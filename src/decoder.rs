//! # Basic decoder block
//! - [X] Optional tril mask
//! - [ ] Positional embeddings
//! - [ ] Multiple hheads

use crate::*;

pub struct DecoderConfig{
    pub tril: bool,
}

impl Default for DecoderConfig{
    fn default() -> Self {
        Self { tril: true }
    }
}

pub struct DecoderBlock{
    wk: Linear,
    wq: Linear,
    wv: Linear,
    proj: Linear,
    conf: DecoderConfig
}

impl DecoderBlock{
    pub fn new(vs: VarBuilder, conf: DecoderConfig) -> Result<Self>{
        Ok(Self {
            conf,
            wk: linear_no_bias(EMBD, HEADD, vs.pp("wk"))?,
            wq: linear_no_bias(EMBD, HEADD, vs.pp("wq"))?,
            wv: linear_no_bias(EMBD, HEADD, vs.pp("wv"))?,
            proj: linear(HEADD, EMBD, vs.pp("proj"))?
        })
    }
}

impl Module for DecoderBlock{
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error>{
        let dev = xs.device();
        let seqd = xs.shape().dims().iter().rev().skip(1).next().unwrap();
        let k = self.wk.forward(xs).unwrap();
        let q = self.wq.forward(xs).unwrap();
        let v = self.wv.forward(xs).unwrap();

        let scores = k.matmul(&q.transpose(D::Minus2, D::Minus1).unwrap()).unwrap();
        let tril = Tensor::tril2(*seqd, DType::F32, dev).unwrap();//.transpose(D::Minus1, D::Minus2).unwrap();

        //println!("{:?}", scores.to_vec2::<f32>());

        let tril = Tensor::from_vec(tril.to_vec2().unwrap().into_iter().flatten().map(|n: f32| if n == 0.{-f32::INFINITY}else{0.} ).collect(), tril.shape(), dev).unwrap();

        let bruh = if self.conf.tril { (tril + scores).unwrap() }else{ scores };
        let scores = softmax(&bruh, D::Minus1).unwrap();
        //println!("bruh: {:?}", bruh.to_vec2::<f32>());
        //println!();
        //println!("scores: {:?}", scores.to_vec2::<f32>());

        // batchd x sqed x headd
        let attn = (scores / (HEADD as f64).sqrt()).unwrap().matmul(&v).unwrap();

        Ok(self.proj.forward(&attn).unwrap())
    }
}

pub fn simple_llm(vs: VarBuilder) -> impl Module{
    (0..3).map(|n| seq().add(DecoderBlock::new(vs.pp(format!("decoder_block_{n}")), DecoderConfig::default()).unwrap()).add(|xs: &Tensor| xs.relu()))
        .fold(seq(), |s, n| s.add(move |xs: &Tensor| (n.forward(xs) + xs).unwrap() * 0.5 ) )
        .add(linear(EMBD, 255, vs.pp("output projection")).unwrap())
}
