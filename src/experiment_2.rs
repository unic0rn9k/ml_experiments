use std::collections::HashMap;

use anyhow::{Error as E, Result, bail};
use candle_core::{Device, DType, Tensor, Module, D};

use candle_transformers::models::bert::{BertModel, Config};

mod token_output_stream;
use token_output_stream::TokenOutputStream;

use candle_nn::{VarBuilder, Linear, linear, VarMap, AdamW, encoding::one_hot, loss::{mse, self}, Optimizer, ops::log_softmax};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use polars::{prelude::*, df};

pub fn main() -> Result<()> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let revision = "refs/pr/21".to_string();

    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        revision,
    ));

    let tokenizer = repo.get("tokenizer.json").unwrap();
    let weights = repo.get("model.safetensors")?;
    let config = repo.get("config.json")?;

    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();

    let mut tokenizer = Tokenizer::from_file(tokenizer).unwrap();

    let config = std::fs::read_to_string(config)?;
    let mut config: Config = serde_json::from_str(&config)?;

    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };

    let model = BertModel::load(vb, &config)?;
    println!("loaded the model in {:?}", start.elapsed());


    let prompt = "All good things come to those who work";

    let tokens = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .unwrap()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    println!("{prompt:?}");

    let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let output = model.forward(&token_ids, &token_type_ids)?.squeeze(0)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let linear = linear(384, 8, vs).unwrap();
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 0.01).unwrap();
    let target = one_hot(Tensor::arange(0i64, 8, &device).unwrap(), 8, 1f32, 0f32).unwrap();

    for epoch in 0..2000{
        //println!("epoch: {epoch}");
        for i in 0..tokens.len(){
            let target = target.get(i)?;

            let out = linear.forward(&output.get(i)?.unsqueeze(0)?).unwrap().squeeze(0).unwrap();

            //println!("{out:?}");
            //println!("{target:?}");

            let loss = mse(&out, &target).unwrap();
            optimizer.backward_step(&loss).unwrap();
        }
    }

    for i in 0..tokens.len(){
        let out: u32 = linear.forward(&output.get(i)?.unsqueeze(0)?).unwrap().squeeze(0).unwrap().argmax(0).unwrap().to_scalar().unwrap();
        println!("{}", out)
    }

    //let reverse_vocab: HashMap<u32, String> = tokenizer.get_vocab(false).into_iter().map(|(tok, id)| (id, tok)).collect();
    //println!("vocab size: {}", reverse_vocab.len());

    //let out = linear.forward(&output)?;
    //for i in 0..tokens.len(){
    //    let token = out.get(i).unwrap().argmax(0).unwrap();
    //    let token = tokenizer.id_to_token(token.to_scalar().unwrap());
    //    println!("{token:?}")
    //}

    //let df = df!{
    //    ""
    //}.unwrap();

    //let mut pipeline = TextGeneration::new(
    //    model,
    //    tokenizer,
    //    args.seed,
    //    args.temperature,
    //    args.top_p,
    //    args.repeat_penalty,
    //    args.repeat_last_n,
    //    &device,
    //);
    //pipeline.run(&args.prompt, args.sample_len)?;

    Ok(())
}
