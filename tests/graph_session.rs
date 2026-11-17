//! Embedding and reranking through a caller-supplied [`GraphSession`], without model weights.
//!
//! A fake session stands in for the ONNX graph and computes deterministic outputs from the token
//! tensors it receives, so each test can derive the expected result from the tokenizer alone.

use fastembed::{
    Error, GraphInput, GraphOutput, GraphSession, OutputKey, Pooling, QuantizationMode,
    TextEmbedding, TextRerank, TokenizerFiles, UserDefinedGraphEmbeddingModel,
    UserDefinedGraphRerankingModel,
};
use ndarray::{Array2, Array3, ArrayD};
use std::sync::{Arc, Mutex};
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    processors::template::TemplateProcessing, Tokenizer,
};

type TestResult = Result<(), tokenizers::Error>;

const PAD: i64 = 0;
const VOCAB: [(&str, u32); 8] = [
    ("[PAD]", 0),
    ("[UNK]", 1),
    ("[CLS]", 2),
    ("[SEP]", 3),
    ("alpha", 4),
    ("beta", 5),
    ("gamma", 6),
    ("delta", 7),
];

/// A word-level tokenizer that wraps inputs in `[CLS] … [SEP]` and marks pair documents with
/// type id 1.
fn tokenizer() -> Result<Tokenizer, tokenizers::Error> {
    let model = WordLevel::builder()
        .vocab(
            VOCAB
                .iter()
                .map(|&(token, id)| (token.into(), id))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")?
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")?
            .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
            .build()?,
    ));
    Ok(tokenizer)
}

fn tokenizer_files() -> Result<TokenizerFiles, tokenizers::Error> {
    Ok(TokenizerFiles {
        tokenizer_file: tokenizer()?.to_string(false)?.into_bytes(),
        config_file: format!(r#"{{"pad_token_id": {PAD}}}"#).into_bytes(),
        special_tokens_map_file: b"{}".to_vec(),
        tokenizer_config_file: br#"{"model_max_length": 512, "pad_token": "[PAD]"}"#.to_vec(),
    })
}

/// What one [`GraphSession::run`] call received.
#[derive(Debug, Clone)]
struct Call {
    input_names: Vec<String>,
    shape: (usize, usize),
    token_type_ids: Option<Array2<i64>>,
}

/// Every call a [`FakeSession`] received, shared with the test.
type Calls = Arc<Mutex<Vec<Call>>>;

type Graph = Box<dyn Fn(&[GraphInput]) -> Vec<GraphOutput> + Send + Sync>;

/// A graph session computing its outputs with `graph` and recording every call.
struct FakeSession {
    input_names: Vec<String>,
    output_names: Vec<String>,
    graph: Graph,
    calls: Calls,
}

impl FakeSession {
    fn boxed(
        input_names: &[&str],
        output_names: &[&str],
        graph: impl Fn(&[GraphInput]) -> Vec<GraphOutput> + Send + Sync + 'static,
    ) -> (Box<dyn GraphSession>, Calls) {
        let calls = Arc::default();
        let session = Self {
            input_names: input_names.iter().map(|name| name.to_string()).collect(),
            output_names: output_names.iter().map(|name| name.to_string()).collect(),
            graph: Box::new(graph),
            calls: Arc::clone(&calls),
        };
        (Box::new(session), calls)
    }
}

impl GraphSession for FakeSession {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, inputs: Vec<GraphInput>) -> fastembed::Result<Vec<GraphOutput>> {
        let ids = input(&inputs, "input_ids");
        self.calls.lock().unwrap().push(Call {
            input_names: inputs.iter().map(|input| input.name.clone()).collect(),
            shape: ids.dim(),
            token_type_ids: inputs
                .iter()
                .find(|input| input.name == "token_type_ids")
                .map(|input| input.tensor.clone()),
        });
        Ok((self.graph)(&inputs))
    }
}

fn input<'a>(inputs: &'a [GraphInput], name: &str) -> &'a Array2<i64> {
    &inputs
        .iter()
        .find(|input| input.name == name)
        .unwrap_or_else(|| panic!("missing input {name}"))
        .tensor
}

/// The fake embedding of the token at `position`, in a row whose unmasked token ids sum to `sum`.
///
/// The first two components differ between padding and real tokens, so mean pooling only matches
/// the reference when the attention mask excludes padding.
/// The row sum makes the `[CLS]` embedding differ between inputs.
fn token_embedding(id: i64, position: usize, sum: i64) -> [f32; 3] {
    [id as f32, (position + 1) as f32, sum as f32]
}

/// The single `last_hidden_state` output of an embedding graph.
fn last_hidden_state(inputs: &[GraphInput]) -> Vec<GraphOutput> {
    vec![hidden_state(inputs)]
}

/// `last_hidden_state` as `[batch, sequence, 3]` token embeddings.
fn hidden_state(inputs: &[GraphInput]) -> GraphOutput {
    let ids = input(inputs, "input_ids");
    let mask = input(inputs, "attention_mask");
    let (batch, sequence) = ids.dim();
    let tensor = Array3::from_shape_fn((batch, sequence, 3), |(row, position, component)| {
        let sum = (0..sequence)
            .map(|p| ids[[row, p]] * mask[[row, p]])
            .sum::<i64>();
        token_embedding(ids[[row, position]], position, sum)[component]
    });
    GraphOutput {
        name: "last_hidden_state".into(),
        tensor: tensor.into_dyn(),
    }
}

fn normalized(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector.iter().map(|value| value / (norm + 1e-12)).collect()
}

/// The embedding of `text` on its own, without padding, computed independently of fastembed.
fn reference_embedding(text: &str, pooling: &Pooling) -> Result<Vec<f32>, tokenizers::Error> {
    let encoding = tokenizer()?.encode(text, true)?;
    let ids = encoding.get_ids().iter().map(|&id| i64::from(id));
    let sum = ids.clone().sum();
    let rows: Vec<[f32; 3]> = ids
        .enumerate()
        .map(|(position, id)| token_embedding(id, position, sum))
        .collect();
    let pooled = match pooling {
        Pooling::Cls => rows[0].to_vec(),
        Pooling::Mean => (0..3)
            .map(|component| rows.iter().map(|row| row[component]).sum::<f32>() / rows.len() as f32)
            .collect(),
    };
    Ok(normalized(&pooled))
}

fn assert_close(have: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(have.len(), want.len());
    for (have, want) in have.iter().zip(want) {
        assert_eq!(have.len(), want.len());
        for (have, want) in have.iter().zip(want) {
            assert!((have - want).abs() < 1e-6, "{have} != {want}");
        }
    }
}

fn embedder(model: UserDefinedGraphEmbeddingModel) -> Result<(TextEmbedding, Calls), Error> {
    let (session, calls) = FakeSession::boxed(
        &["input_ids", "attention_mask"],
        &["last_hidden_state"],
        last_hidden_state,
    );
    Ok((
        TextEmbedding::try_new_from_graph_session(session, model)?,
        calls,
    ))
}

const TEXTS: [&str; 5] = [
    "alpha",
    "beta gamma delta alpha",
    "gamma",
    "delta delta",
    "alpha beta gamma",
];

/// CLS and mean pooling plus L2 normalization run on the session's token embeddings.
#[test]
fn pools_and_normalizes_session_outputs() -> TestResult {
    for pooling in [Pooling::Cls, Pooling::Mean] {
        let (mut model, calls) = embedder(
            UserDefinedGraphEmbeddingModel::new(tokenizer_files()?).with_pooling(pooling.clone()),
        )?;
        let have = model.embed(TEXTS, None)?;
        let want = TEXTS
            .iter()
            .map(|text| reference_embedding(text, &pooling))
            .collect::<Result<Vec<_>, _>>()?;

        // One batch padded to the longest input, here six tokens with `[CLS]` and `[SEP]`.
        assert_eq!(
            calls
                .lock()
                .unwrap()
                .iter()
                .map(|call| call.shape)
                .collect::<Vec<_>>(),
            [(5, 6)]
        );
        // Every input gets the embedding it has on its own, so padding is masked out.
        assert_close(&have, &want);
    }
    Ok(())
}

/// Several batches of mixed lengths return embeddings in input order.
#[test]
fn batches_preserve_input_order() -> TestResult {
    let (mut model, calls) = embedder(
        UserDefinedGraphEmbeddingModel::new(tokenizer_files()?).with_pooling(Pooling::Mean),
    )?;
    let have = model.embed(TEXTS, Some(2))?;
    let want = TEXTS
        .iter()
        .map(|text| reference_embedding(text, &Pooling::Mean))
        .collect::<Result<Vec<_>, _>>()?;

    // Each batch is padded only to its own longest input.
    assert_eq!(
        calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| call.shape)
            .collect::<Vec<_>>(),
        [(2, 6), (2, 4), (1, 5)]
    );
    assert_close(&have, &want);
    Ok(())
}

/// Fixed padding pins every batch to the profile length, without changing the embeddings.
#[test]
fn fixed_padding_applies_to_graph_sessions() -> TestResult {
    let (mut model, calls) = embedder(
        UserDefinedGraphEmbeddingModel::new(tokenizer_files()?).with_pooling(Pooling::Mean),
    )?;
    model.set_fixed_padding(8)?;
    let have = model.embed(TEXTS, Some(2))?;
    let want = TEXTS
        .iter()
        .map(|text| reference_embedding(text, &Pooling::Mean))
        .collect::<Result<Vec<_>, _>>()?;

    assert_eq!(
        calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| call.shape)
            .collect::<Vec<_>>(),
        [(2, 8), (2, 8), (1, 8)]
    );
    assert_close(&have, &want);
    Ok(())
}

/// Dynamically quantized models run every input in one batch and refuse smaller batches.
#[test]
fn dynamic_quantization_runs_one_batch() -> TestResult {
    let (mut model, calls) = embedder(
        UserDefinedGraphEmbeddingModel::new(tokenizer_files()?)
            .with_quantization(QuantizationMode::Dynamic),
    )?;
    model.embed(TEXTS, None)?;
    assert_eq!(calls.lock().unwrap().len(), 1);

    // A batch smaller than the input is refused before the session runs.
    assert!(matches!(
        model.embed(TEXTS, Some(2)),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(calls.lock().unwrap().len(), 1);
    Ok(())
}

/// The session receives `token_type_ids` exactly when it declares that input.
#[test]
fn token_type_ids_follow_declared_inputs() -> TestResult {
    for declared in [false, true] {
        let input_names: &[&str] = if declared {
            &["input_ids", "attention_mask", "token_type_ids"]
        } else {
            &["input_ids", "attention_mask"]
        };
        let (session, calls) =
            FakeSession::boxed(input_names, &["last_hidden_state"], last_hidden_state);
        let mut model = TextEmbedding::try_new_from_graph_session(
            session,
            UserDefinedGraphEmbeddingModel::new(tokenizer_files()?),
        )?;
        model.embed(["alpha beta"], None)?;

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].input_names, input_names);
        // A single sentence is one segment, so every type id is zero.
        if declared {
            assert_eq!(calls[0].token_type_ids, Some(Array2::zeros((1, 4))));
        }
    }
    Ok(())
}

/// Default precedence picks `last_hidden_state` among several outputs; an output key overrides it.
#[test]
fn output_key_selects_among_named_outputs() -> TestResult {
    let sentence_embedding = |inputs: &[GraphInput]| {
        let batch = input(inputs, "input_ids").nrows();
        GraphOutput {
            name: "sentence_embedding".into(),
            tensor: ArrayD::from_shape_fn(vec![batch, 2], |index| {
                (index[0] + 3 * index[1] + 1) as f32
            }),
        }
    };
    let build = |output_key: Option<OutputKey>| -> Result<TextEmbedding, tokenizers::Error> {
        let (session, _) = FakeSession::boxed(
            &["input_ids", "attention_mask"],
            &["sentence_embedding", "last_hidden_state"],
            move |inputs| vec![sentence_embedding(inputs), hidden_state(inputs)],
        );
        let mut model =
            UserDefinedGraphEmbeddingModel::new(tokenizer_files()?).with_pooling(Pooling::Mean);
        if let Some(output_key) = output_key {
            model = model.with_output_key(output_key);
        }
        Ok(TextEmbedding::try_new_from_graph_session(session, model)?)
    };

    // Precedence skips `OnlyOne` with two outputs and falls through to `last_hidden_state`.
    let have = build(None)?.embed(["alpha", "beta gamma"], None)?;
    let want = ["alpha", "beta gamma"]
        .iter()
        .map(|text| reference_embedding(text, &Pooling::Mean))
        .collect::<Result<Vec<_>, _>>()?;
    assert_close(&have, &want);

    // An already pooled output is only normalized.
    let have = build(Some(OutputKey::ByName("sentence_embedding")))?
        .embed(["alpha", "beta gamma"], None)?;
    assert_close(&have, &[normalized(&[1.0, 4.0]), normalized(&[2.0, 5.0])]);
    Ok(())
}

/// Outputs that differ from the declared names are rejected rather than selected by position.
#[test]
fn outputs_must_match_declared_names() -> TestResult {
    let (session, _) = FakeSession::boxed(
        &["input_ids", "attention_mask"],
        &["last_hidden_state"],
        |inputs| {
            let mut output = hidden_state(inputs);
            output.name = "token_embeddings".into();
            vec![output]
        },
    );
    let mut model = TextEmbedding::try_new_from_graph_session(
        session,
        UserDefinedGraphEmbeddingModel::new(tokenizer_files()?),
    )?;
    assert!(matches!(
        model.embed(["alpha"], None),
        Err(Error::GraphSession(_))
    ));
    Ok(())
}

/// Only in-process ONNX Runtime sessions report CUDA allocator statistics.
#[test]
fn cuda_allocator_stats_need_an_in_process_session() -> TestResult {
    let (model, _) = embedder(UserDefinedGraphEmbeddingModel::new(tokenizer_files()?))?;
    assert!(matches!(model.cuda_allocator_stats(), Err(Error::Other(_))));

    let (session, _) = FakeSession::boxed(&["input_ids", "attention_mask"], &["logits"], logits);
    let rerank = TextRerank::try_new_from_graph_session(
        session,
        UserDefinedGraphRerankingModel::new(tokenizer_files()?),
    )?;
    assert!(matches!(
        rerank.cuda_allocator_stats(),
        Err(Error::Other(_))
    ));
    Ok(())
}

/// The relevance of one token: `alpha` 1, `beta` 2, `gamma` 3, anything else 0.
fn weight(id: i64) -> f32 {
    match id {
        4 => 1.0,
        5 => 2.0,
        6 => 3.0,
        _ => 0.0,
    }
}

/// `[batch, 2]` logits: the summed token weights of each pair, then their negation as a decoy.
fn logits(inputs: &[GraphInput]) -> Vec<GraphOutput> {
    let ids = input(inputs, "input_ids");
    let mask = input(inputs, "attention_mask");
    let tensor = Array2::from_shape_fn((ids.nrows(), 2), |(row, column)| {
        let score = ids
            .row(row)
            .iter()
            .zip(mask.row(row))
            .map(|(&id, &mask)| weight(id) * mask as f32)
            .sum::<f32>();
        if column == 0 {
            score
        } else {
            -score
        }
    });
    vec![GraphOutput {
        name: "logits".into(),
        tensor: tensor.into_dyn(),
    }]
}

/// Scores come from the first logit column and sort descending, with ties kept in input order.
#[test]
fn rerank_scores_sort_by_first_logit_column() -> TestResult {
    let (session, calls) = FakeSession::boxed(
        &["input_ids", "attention_mask", "token_type_ids"],
        &["logits"],
        logits,
    );
    let mut model = TextRerank::try_new_from_graph_session(
        session,
        UserDefinedGraphRerankingModel::new(tokenizer_files()?),
    )?;
    let documents = ["beta", "gamma", "beta", "delta", "gamma beta"];
    let results = model.rerank("alpha", documents, true, Some(2))?;

    // With the query's weight of 1 the scores are 3, 4, 3, 1 and 6; the tied documents 0 and 2
    // keep their input order.
    assert_eq!(
        results
            .iter()
            .map(|result| result.index)
            .collect::<Vec<_>>(),
        [4, 1, 0, 2, 3]
    );
    assert_eq!(
        results
            .iter()
            .map(|result| result.score)
            .collect::<Vec<_>>(),
        [6.0, 4.0, 3.0, 3.0, 1.0]
    );
    assert_eq!(
        results
            .iter()
            .map(|result| result.document.as_deref())
            .collect::<Vec<_>>(),
        [
            Some("gamma beta"),
            Some("gamma"),
            Some("beta"),
            Some("beta"),
            Some("delta")
        ]
    );

    // Three batches ran, and pair documents carry type id 1: `[CLS] alpha [SEP] beta [SEP]`.
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|call| call.shape).collect::<Vec<_>>(),
        [(2, 5), (2, 5), (1, 6)]
    );
    assert_eq!(
        calls[0]
            .token_type_ids
            .as_ref()
            .map(|ids| ids.row(0).to_vec()),
        Some(vec![0, 0, 0, 1, 1])
    );
    Ok(())
}

/// A reranker session that does not declare `token_type_ids` does not receive it.
#[test]
fn rerank_omits_undeclared_token_type_ids() -> TestResult {
    let (session, calls) =
        FakeSession::boxed(&["input_ids", "attention_mask"], &["logits"], logits);
    let mut model = TextRerank::try_new_from_graph_session(
        session,
        UserDefinedGraphRerankingModel::new(tokenizer_files()?),
    )?;
    model.rerank("alpha", ["beta"], false, None)?;
    assert_eq!(
        calls.lock().unwrap()[0].input_names,
        ["input_ids", "attention_mask"]
    );
    Ok(())
}

/// Logits without one row per pair are an error, not a panic or misaligned scores.
#[test]
fn rerank_rejects_misshaped_logits() -> TestResult {
    let shapes: [&[usize]; 3] = [&[3], &[3, 2], &[2, 0]];
    for shape in shapes {
        let shape = shape.to_vec();
        let (session, _) =
            FakeSession::boxed(&["input_ids", "attention_mask"], &["logits"], move |_| {
                vec![GraphOutput {
                    name: "logits".into(),
                    tensor: ArrayD::zeros(shape.clone()),
                }]
            });
        let mut model = TextRerank::try_new_from_graph_session(
            session,
            UserDefinedGraphRerankingModel::new(tokenizer_files()?),
        )?;
        assert!(matches!(
            model.rerank("alpha", ["beta", "gamma"], false, None),
            Err(Error::InvalidShape(_))
        ));
    }
    Ok(())
}
