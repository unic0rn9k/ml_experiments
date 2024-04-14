use std::iter::repeat_with;
use std::{fs::File, io::BufRead};
use std::io::BufReader;
use std::ops::Deref;

use anyhow::Result;
use candle_nn::encoding::one_hot;
use candle_nn::ops::log_softmax;
use candle_nn::{*, ops::softmax};
use candle_core::*;

mod decoder;
mod experiment_1;
mod experiment_2;
mod experiment_3;
use decoder::*;

const EMBD: usize = 100;
const HEADD: usize = 300;
const NHEADS: usize = 4;

macro_rules! projects {
    ($($name:ident),*) => {
        const PROJECTS: &[&str] = &[$(stringify!($name)),*];

        match std::env::args().skip(1).next().as_ref().map(Deref::deref){
            $(Some(stringify!($name)) => $name::main().unwrap(),)*
            name => panic!("Expected one argument containing name of project. Found: {name:?}. Available options are {PROJECTS:?}."),
        }
    };
}

fn main(){
    projects!(decoder, experiment_1, experiment_2, experiment_3);
}

pub fn argmax(x: &[f32]) -> usize{
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
