use super::*;
use crate::test_support::{word_level_tokenizer_files, ScriptedSession};
use ndarray::{arr2, Array3, ArrayD};

/// A sparse model of `output` whose graph returns `tensor` as its only output.
fn scripted(
    output: SparseOutput,
    tensor: ArrayD<f32>,
) -> Result<SparseTextEmbedding<ScriptedSession>> {
    SparseTextEmbedding::try_new_from_graph_session(
        ScriptedSession::returning("logits", tensor),
        UserDefinedGraphSparseModel::new(word_level_tokenizer_files(), output),
    )
}

/// Two texts of one and two tokens, so the first row of the batch is padded.
const TEXTS: [&str; 2] = ["hello", "hello world"];

/// Deterministic logits in `[-4, 4)`, without a random number generator.
fn logits(shape: (usize, usize, usize)) -> Array3<f32> {
    let mut state = 0x2545_f491_u32;
    Array3::from_shape_simple_fn(shape, || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32 * 8.0 - 4.0
    })
}

#[test]
fn splade_post_processing_max_pools_over_unmasked_tokens_only() {
    // batch=1, seq=3, vocab=4. The last position is padding and carries the largest logits
    let logits = Array3::from_shape_vec(
        (1, 3, 4),
        vec![
            1.0, -1.0, 0.0, 0.5, //
            0.5, 2.0, -3.0, 0.0, //
            9.0, 9.0, 9.0, 9.0,
        ],
    )
    .unwrap();
    let mask = arr2(&[[1i64, 1, 0]]);

    let out = post_process_splade(&logits.view().into_dyn(), &mask);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].indices, vec![0, 1, 3]);
    let expected = [2.0f32.ln(), 3.0f32.ln(), 1.5f32.ln()];
    for (value, expected) in out[0].values.iter().zip(expected) {
        assert!((value - expected).abs() < 1e-6, "{value} != {expected}");
    }
}

/// A pooled model's weights are the graph's own: the pipeline selects each row's nonzero entries
/// and applies no activation of its own on top.
#[test]
fn pooled_splade_keeps_the_weights_the_graph_returns() -> Result<()> {
    let pooled = arr2(&[[0.0_f32, 0.25, 0.0, 3.0], [1.5, 0.0, 0.0, 0.0]]).into_dyn();
    let mut model = scripted(SparseOutput::PooledSplade, pooled)?;

    let embeddings = model.embed(TEXTS, None)?;

    assert_eq!(
        embeddings,
        vec![
            SparseEmbedding {
                indices: vec![1, 3],
                values: vec![0.25, 3.0],
            },
            SparseEmbedding {
                indices: vec![0],
                values: vec![1.5],
            },
        ]
    );
    Ok(())
}

/// A pooled model refuses per-token logits, so it never pools them, and an unpooled SPLADE model
/// refuses pooled weights, so it never pools them a second time.
#[test]
fn sparse_models_refuse_an_output_of_the_other_rank() -> Result<()> {
    let token_logits = logits((2, 2, 4)).into_dyn();
    let pooled = arr2(&[[0.0_f32, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).into_dyn();

    for (output, tensor) in [
        (SparseOutput::PooledSplade, token_logits),
        (SparseOutput::Splade, pooled),
    ] {
        let mut model = scripted(output, tensor)?;
        let result = model.embed(TEXTS, None);
        assert!(
            matches!(result, Err(Error::InvalidShape(_))),
            "{output:?}: {result:?}"
        );
    }
    Ok(())
}

/// Pooled weights with a row count other than the batch's would misalign embeddings and texts.
#[test]
fn pooled_splade_refuses_rows_that_do_not_match_the_batch() -> Result<()> {
    let three_rows = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]]).into_dyn();
    let mut model = scripted(SparseOutput::PooledSplade, three_rows)?;

    let result = model.embed(TEXTS, None);

    assert!(matches!(result, Err(Error::InvalidShape(_))), "{result:?}");
    Ok(())
}

/// Pooling in the graph, as the pooled bundle does it, then selecting the nonzero weights gives
/// what pooling the token logits here gives, padded positions included.
#[test]
fn pooling_in_the_graph_matches_pooling_token_logits() {
    let logits = logits((3, 5, 64));
    // A short, a full and a single-token sequence; the padded positions carry the logits a
    // masked token must never contribute.
    let mask = arr2(&[[1_i64, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]);
    let graph_pooled = Array2::from_shape_fn((3, 64), |(row, vocab)| {
        (0..5)
            .map(|position| {
                let activated = logits[[row, position, vocab]].max(0.0).ln_1p();
                activated * mask[[row, position]] as f32
            })
            .fold(f32::MIN, f32::max)
    });

    let pooled_here = post_process_splade(&logits.view().into_dyn(), &mask);
    let pooled_in_graph = post_process_pooled_splade(&graph_pooled.view());

    assert_eq!(pooled_in_graph.len(), pooled_here.len());
    for (in_graph, here) in pooled_in_graph.iter().zip(&pooled_here) {
        assert_eq!(in_graph.indices, here.indices);
        for (graph_value, value) in in_graph.values.iter().zip(&here.values) {
            assert!(
                (graph_value - value).abs() < 1e-5,
                "{graph_value} != {value}"
            );
        }
    }
}
