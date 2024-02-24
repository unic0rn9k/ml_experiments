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

const EMBD: usize = 100;
const HEADD: usize = 300;
const NHEADS: usize = 4;

fn main() -> Result<()>{
    let (decoder, vars) = experiment_1::simple_llm();
    let dev = Device::Cpu;

    let mut opt = AdamW::new_lr(vars, 0.01)?;

    let mut file = BufReader::new(File::open("tiny-shakespeare.txt")?).lines();
    let mut cost_sum = 0.;
    for n in 0..{
        let line = file.next().expect("EOF")?;
        let tokens1 = line.as_bytes();

        let line = file.next().expect("EOF")?;
        let tokens2 = line.as_bytes();
        let seqdy = tokens2.len();
        let seqdx = tokens1.len();

        if seqdx < 2 || seqdy < 2{
            continue
        }

        let ys = Tensor::new(tokens2, &dev)?;
        let tokens = Tensor::new(tokens1, &dev)?;

        //let ys = one_hot(ys, 255, 1., 0.).unwrap();

        let yhat = (decoder)(&tokens, seqdy);

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
