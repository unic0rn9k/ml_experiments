use std::iter::repeat_with;
use std::{fs::File, io::BufRead};
use std::io::BufReader;

use anyhow::Result;
use candle_nn::encoding::one_hot;
use candle_nn::ops::log_softmax;
use candle_nn::{*, ops::softmax};
use candle_core::*;

mod decoder;
mod experiment_1;
use decoder::*;

const EMBD: usize = 200;
const HEADD: usize = 300;
const NHEADS: usize = 4;

fn main() -> Result<()>{
    let dev = Device::Cpu;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let decoder = simple_llm(vs.clone());

    let embeddings = Var::randn(0f32, 1f32, (255, EMBD), &dev)?;
    
    let mut vars = varmap.all_vars();
    vars.push(embeddings.clone());

    let mut opt = AdamW::new_lr(vars, 0.01)?;

    let mut file = BufReader::new(File::open("tiny-shakespeare.txt")?).lines();
    let mut cost_sum = 0.;
    for n in 0..{
        let line = file.next().expect("EOF")?;
        let tokens = line.as_bytes();
        let seqd = tokens.len();

        if seqd < 2{
            continue
        }

        let ys = Tensor::new(&tokens[1..seqd], &dev)?;
        let tokens = Tensor::new(&tokens[0..seqd-1], &dev)?;

        //let ys = one_hot(ys, 255, 1., 0.).unwrap();

        let xs = embeddings.as_tensor().embedding(&tokens)?;
        let yhat = decoder.forward(&xs).unwrap();

        let log_sm = log_softmax(&yhat, D::Minus1)?;

        let loss = loss::nll(&log_sm, &ys).unwrap();
        cost_sum += loss.to_scalar::<f32>().unwrap();
        opt.backward_step(&loss).unwrap();

        if n % 100 == 0{
            println!("{n}: {}", cost_sum/n as f32);
            for row in yhat.to_vec2().unwrap(){
                print!("{}", argmax(&row) as u8 as char)
            }
            println!();
            cost_sum = 0.;
        }
    }

    Ok(())
}

fn argmax(x: &[f32]) -> usize{
    let mut i = 0;
    for (j, k) in x.iter().copied().enumerate(){
        if k > x[i]{
            i = j
        }
    }
    i
}

#[test]
fn bob(){
    let dev = Device::Cpu;

    let a = Tensor::from_slice(&[1.,2., 3.,4.], (2,2), &dev).unwrap();
    let b = Tensor::from_slice(&[1.,1., 1.,1.], (2,2), &dev).unwrap();

    println!("{:?}", a.broadcast_mul(&b).unwrap().to_vec2::<f64>());
}
