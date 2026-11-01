use katu_core::{penalize_repetitions, GenerationOptions, KatuLmConfig};
#[test]
fn migration_preserves_old_logits_and_weights() {
    let c = KatuLmConfig {
        vocab_size: 32,
        block_size: 16,
        embed_dim: 8,
        layers: 1,
        heads: 2,
    };
    let p = c.init(1).unwrap();
    let (new, q) = c.extend_context(&p, 256, 42).unwrap();
    assert_eq!(q[1].shape(), &[256, 8]);
    assert_eq!(&q[1].data()[..p[1].numel()], p[1].data());
    for i in 0..p.len() {
        if i != 1 {
            assert_eq!(p[i].data(), q[i].data());
        }
    }
    assert_eq!(
        c.forward(&p, &[1, 2, 3]).unwrap().data(),
        new.forward(&q, &[1, 2, 3]).unwrap().data()
    );
    assert!(c.extend_context(&p, 8, 42).is_err());
}
#[test]
fn context_migration_does_not_share_accumulated_gradients() {
    let c = KatuLmConfig {
        vocab_size: 8,
        block_size: 4,
        embed_dim: 4,
        layers: 0,
        heads: 1,
    };
    let p = c.init(1).unwrap();
    c.loss(&p, &[&[1, 2, 3]]).unwrap().backward().unwrap();
    assert!(p[0].grad().unwrap().is_some());
    let (_, q) = c.extend_context(&p, 8, 2).unwrap();
    assert!(q.iter().all(|p| p.grad().unwrap().is_none()));
}

#[test]
fn repetition_penalty_has_correct_sign_and_no_double_counting() {
    let mut v = vec![2., -2., 3.];
    penalize_repetitions(&mut v, &[0, 0, 1], 2.).unwrap();
    assert_eq!(v, vec![1., -4., 3.]);
    assert!(penalize_repetitions(&mut v, &[], f32::NAN).is_err());
    assert!(penalize_repetitions(&mut v, &[99], 1.2).is_err());
}
#[test]
fn stop_sequence_ends_generation() {
    let c = KatuLmConfig {
        vocab_size: 4,
        block_size: 4,
        embed_dim: 4,
        layers: 0,
        heads: 1,
    };
    let p = c.init(1).unwrap();
    let first = c.generate(&p, &[0], 1, 0., 1).unwrap()[1];
    let result = c
        .generate_with_options(
            &p,
            &[0],
            10,
            &GenerationOptions {
                temperature: 0.,
                repetition_penalty: 1.,
                seed: 1,
                stop_tokens: vec![first],
            },
        )
        .unwrap();
    assert_eq!(result.len(), 2);
}
