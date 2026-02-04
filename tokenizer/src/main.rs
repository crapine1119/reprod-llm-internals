use std::env;

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::{ModelWrapper, TrainerWrapper};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::tokenizer::SplitDelimiterBehavior;
use tokenizers::{AddedToken, Result, Tokenizer};

fn main() -> Result<()> {
    // 사용법:
    // cargo run -- <corpus_with_mecab_tags.txt> <out_tokenizer.json> <vocab_size> <min_frequency>
    //
    // 예)
    // cargo run -- asset/mecab_ko.txt asset/tokenizer.json 32000 2

    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "Usage: {} <corpus.txt> <out_tokenizer.json> <vocab_size> <min_frequency>",
            args[0]
        );
        std::process::exit(2);
    }

    let corpus_path = args[1].clone();
    let out_path = args[2].clone();
    let vocab_size: usize = args[3].parse()?;
    let min_frequency: u64 = args[4].parse()?;

    // 1) BPE 모델 생성
    let bpe = BPE::builder().unk_token("<unk>".into()).build()?;

    // 2) Tokenizer 생성
    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));

    // 3) PreTokenizer:
    //    - 공백 분리
    //    - "<mecab>" 기준 split, delimiter는 제거(Removed)
    let mecab_split = Split::new("<mecab>", SplitDelimiterBehavior::Removed, false)?;
    let pre_tokenizer = Sequence::new(vec![
        WhitespaceSplit.into(),
        mecab_split.into(),
    ]);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    // 4) BPE Trainer 생성
    let bpe_trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(min_frequency)
        .special_tokens(vec![
            AddedToken::from("<s>", true),
            AddedToken::from("<pad>", true),
            AddedToken::from("</s>", true),
            AddedToken::from("<unk>", true),
            AddedToken::from("<mask>", true),
        ])
        .build();

    // 5) 핵심: BpeTrainer -> TrainerWrapper 로 감싸기
    //    TrainerWrapper는 BpeTrainer variant를 갖고, Trainer<Model=ModelWrapper> 입니다. :contentReference[oaicite:3]{index=3}
    let mut trainer = TrainerWrapper::from(bpe_trainer);

    // 6) 학습 + 저장
    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec![corpus_path])?
        .save(out_path, pretty)?;

    Ok(())
}
