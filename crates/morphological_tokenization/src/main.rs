use ml_experiments::decoder::simple_llm;
use morphological_tokenization::*;
use ml_experiments::*;
use std::collections::HashMap;
use std::io::Read;

use std::iter::repeat_with;
use std::{fs::File, io::BufRead};
use std::io::BufReader;
use std::ops::Deref;

use candle_nn::encoding::one_hot;
use candle_nn::ops::log_softmax;
use candle_nn::{*, ops::softmax};
use candle_core::*;

const EMBD: usize = 64;
const HEADD: usize = 32;
const NHEADS: usize = 4;

// # Performance
// after 20 iterations: 4993
// Naive 3 heads: 4188/41min
// Relu on risidual stream: 4300/23min
// 3980 - 2 hours 10min
// smaller model: 3000/30sec

pub fn pos_embeddings(n: usize) -> Tensor{
    let mut tmp = vec![];

    for y in 0..EMBD{
        for x in 0..n{
            tmp.push((y as f32 * x as f32).sin())
        }
    }

    Tensor::from_vec(tmp, (1, n, EMBD), &Device::Cpu).unwrap()
}

fn sample(model: &impl Module, n: usize, dev: &Device, embeddings: &Var) -> String{
    let mut tokens = vec![0, 3]; // '.' token

    for n in 1..n{
        let seqd = n*2;
        let toks = Tensor::new(tokens.clone(), dev).unwrap();

        let embeddings = (embeddings.as_tensor().embedding(&toks).unwrap().reshape((1, seqd, EMBD)) + pos_embeddings(seqd)).unwrap();
        let ys = model.forward(&embeddings).unwrap();

        let ys = &ys.to_vec3().unwrap()[0];

        let [a, b] = &ys[ys.len()-2..ys.len()] else {unreachable!()};
        tokens.push(argmax(&a) as u32);
        tokens.push(argmax(&b) as u32);
    }

    decode(&tokens)
}

pub fn main() -> Result<()>{
    if std::env::args().skip(1).next() == Some("sample".into()){
        let dev = Device::Cpu;
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let decoder = simple_llm(vs.clone(), *TOKENN as usize);
        let embeddings = Var::randn(0f32, 1f32, (*TOKENN as usize, EMBD), &dev)?;
    
        varmap.load("bruh6.safetensors").unwrap();

        println!("{}", sample(&decoder, 200, &dev, &embeddings));
    }

    let dev = Device::Cpu;
    let tril = Tensor::tril2(5, DType::F32, &dev).unwrap();
    println!("{tril}");

    let batch_size = 30;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let decoder = simple_llm(vs.clone(), *TOKENN as usize);

    let embeddings = Var::randn(0f32, 1f32, (*TOKENN as usize, EMBD), &dev)?;
    let pos_embeddings: Vec<_> = repeat_with(|| Var::randn(0f32, 1f32, (1, 1, EMBD), &dev).unwrap() ).collect();
    //let target_embeddings = Var::randn(0f32, 1f32, (1, 2, EMBD), &dev)?;
    
    let mut vars = varmap.all_vars();
    vars.push(embeddings.clone());
    //vars.push(target_embeddings.clone());
    vars.append(&mut pos_embeddings.clone());
    let pos_embeddings: Vec<_> = pos_embeddings.iter().map(|n|n.as_tensor()).collect();


    let mut opt = AdamW::new_lr(vars, 0.01)?;

    let mut file = File::open("bruh.txt")?.bytes();
    let mut cost_sum = 0.;
    for n in 0..{
        let seqd = (n*2) % 40 + 2;

        let mut ys = vec![];
        let mut xs = vec![];
        let mut txt = HashMap::<usize, String>::new();

        for _n in 0..batch_size{
            let file = repeat_with(||file.next().expect("EOF"));
            let tokens: Vec<u32> = TokenizerStream::new(
                &mut file
                    .map(|c|c.unwrap() as char)
                    .map(|c| {txt.entry(_n).or_default().push(c); c})
            ).take(seqd+2).collect();

            ys.push(Tensor::new(&tokens[2..seqd+2], &dev)?);

            let toks = Tensor::new(&tokens[0..seqd-2], &dev)?;
            let pos_embeddings = Tensor::cat(&pos_embeddings[0..seqd], D::Minus2).unwrap();
            let toks = ( embeddings.as_tensor().embedding(&toks).unwrap().reshape((1, seqd, EMBD)) + pos_embeddings).unwrap();

            xs.push(toks);
        }

        if seqd < 2{
            unreachable!()
        }

        let xs = Tensor::cat(&xs, 0).unwrap();
        //let ys = Tensor::cat(&ys, 0).unwrap();

        let yhat = decoder.forward(&xs).unwrap();

        //let yhat = yhat.reshape((batch_size, 2, *TOKENN as usize)).unwrap();
        let log_sm = log_softmax(&yhat, D::Minus1)?;

        let mut loss = Tensor::new(0f32, &dev).unwrap();
        for n in 0..batch_size{
            loss = (loss + loss::nll(&log_sm.get_on_dim(0, n).unwrap(), &ys[n]).unwrap()).unwrap();
            cost_sum += loss.to_scalar::<f32>().unwrap();
        }
        loss = (loss / Tensor::new(batch_size as f32, &dev)).unwrap();

        opt.backward_step(&loss).unwrap();
        {
            println!("{n}: {}", cost_sum/n as f32);
            let mut buffer = vec![];
            for row in &yhat.to_vec3().unwrap()[0]{
                buffer.push(argmax(&row) as u32)
            }
            println!("{}", decode(&buffer));
            if n % 10 == 0{
                varmap.save("bruh6.safetensors").unwrap();
            }
        }
    }

    Ok(())
}
