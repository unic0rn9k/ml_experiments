//! # Meta
//! - created : 05. Feb 22:48 - 2024
//!
//! # Concept
//! 0. decoder block
//! 1. Sum across tokens
//! 2. create one big vectror
//! 3. chop ino tokens
//! 4. repeat prev n times
//! 5. duplicate for each output token.
//! 6. add position-encoding
//! 7. final decoder again
//!
//! # TODO
//! - [ ] Softmax after every linear tokenizer

const RECODERD: usize = 600;
const RECODERS: usize = 20;

use crate::*;

pub struct RecoderBlock {
    tokenizer: Vec<Linear>,
    pos: Var,
    decoder: DecoderBlock,
    proj: Linear,
}

impl RecoderBlock {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            tokenizer: (0..RECODERS).map(|n| linear(RECODERD, EMBD, vs.pp(format!("recoder_token_{n}"))).unwrap() ).collect(),
            pos: Var::randn(0f32, 1f32, (RECODERS, EMBD), vs.device()).unwrap(),
            decoder: DecoderBlock::new(vs.pp("decoder")).unwrap(), // TODO: Optionally ommit tril-mask
            proj: linear(EMBD, RECODERD, vs.pp("recoder_out_projection")).unwrap(),
        })
    }
}

impl Module for RecoderBlock{
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // TODO: self.pos
        let toks: Vec<_> = self.tokenizer.iter().enumerate().map(|(n, tok)| tok.forward(xs).unwrap()).collect();
        let xs = Tensor::cat(&toks, 0).unwrap();
        let dec = self.decoder.forward(&xs).unwrap();
        let blown = self.proj.forward(&dec).unwrap();

        let ret = blown.sum_keepdim(D::Minus2).unwrap();
        Ok(ret)
    }
}

pub fn simple_llm() -> (impl Fn(&Tensor, usize)->Tensor, Vec<Var>){
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let embeddings = Var::randn(0f32, 1f32, (255, EMBD), &dev).unwrap();

    // TODO: Risidual connection.
    let llm = (0..3).map(|n| seq().add(RecoderBlock::new(vs.pp(format!("recoder_block_{n}"))).unwrap()).add(|xs: &Tensor| xs.tanh()))
        .fold(seq(), |s, n| s.add(move |xs: &Tensor| n.forward(xs)));

    let wtf: Vec<_> = (0..300).map(|n| linear(RECODERD, EMBD, vs.pp(format!("wtf_{n}"))).unwrap()).collect();
    let unwtf = linear(EMBD, RECODERD, vs.pp("unwtf")).unwrap();
    let out_decoder = DecoderBlock::new(vs.pp("final_output_decoder")).unwrap();
    let fin_tok = linear(EMBD, 255, vs.pp("fintok")).unwrap();

    let mut vars = varmap.all_vars();
    vars.push(embeddings.clone());

    let decoder_in = DecoderBlock::new(vs.pp("initial_input_decoder")).unwrap();

    (
        move |xs: &Tensor, seqdy|{
            let xs = embeddings.embedding(xs).unwrap();
            let xs = decoder_in.forward(&xs).unwrap();
            let xs = unwtf.forward(&xs).unwrap();
            let xs = xs.sum_keepdim(D::Minus2).unwrap();
            let out = llm.forward(&xs).unwrap();
            let toks: Vec<_> = wtf.iter().take(seqdy).map(|tok| tok.forward(&out).unwrap() ).collect();
            let toks = Tensor::cat(&toks[..], 0).unwrap();
            let out = out_decoder.forward(&toks).unwrap();
            fin_tok.forward(&out).unwrap()
        },
        vars
    )
}
