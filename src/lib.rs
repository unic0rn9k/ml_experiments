use std::iter::repeat_with;
use std::{fs::File, io::BufRead};
use std::io::BufReader;

use anyhow::Result;
use candle_nn::encoding::one_hot;
use candle_nn::ops::log_softmax;
use candle_nn::{*, ops::softmax};
use candle_core::*;

const EMBD: usize = 100;
const HEADD: usize = 300;
const NHEADS: usize = 4;

pub mod decoder;
pub use candle_core;
pub use candle_nn;
pub use rust_bert;
pub use indicatif;
pub use ::safetensors;
pub use rayon;
pub use memmap2;

pub fn argmax(x: &[f32]) -> usize{
    let mut i = 0;
    for (j, k) in x.iter().copied().enumerate(){
        if k > x[i]{
            i = j
        }
    }
    i
}
