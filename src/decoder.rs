//! # Basic decoder block
//! - [X] Optional tril mask
//! - [ ] Positional embeddings
//! - [-] Batching
//! - [ ] Multiple hheads

use std::io::Read;

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

    /// batch x head x seqd x embd
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error>{
        let dev = xs.device();
        let [batch_size, seqd, embdd] = xs.dims() else {panic!("Too many dims")};
        assert_eq!(*embdd, EMBD);
        let k = self.wk.forward(xs).unwrap();
        let q = self.wq.forward(xs).unwrap();
        let v = self.wv.forward(xs).unwrap();

        let scores = k.matmul(&q.transpose(D::Minus2, D::Minus1).unwrap()).unwrap();
        let tril = Tensor::tril2(*seqd, DType::F32, dev).unwrap();//.transpose(D::Minus1, D::Minus2).unwrap();

        //println!("{:?}", scores.to_vec2::<f32>());

        let tril = Tensor::from_vec(tril.to_vec2().unwrap().into_iter().flatten().map(|n: f32| if n == 0.{-f32::INFINITY}else{0.} ).collect(), tril.shape(), dev).unwrap();

        let bruh = if self.conf.tril { (tril.repeat((*batch_size, 1, 1)) + scores).unwrap() }else{ scores };
        let scores = softmax(&bruh, D::Minus1).unwrap();
        //println!("bruh: {:?}", bruh.to_vec2::<f32>());
        //println!();
        //println!("scores: {:?}", scores.to_vec2::<f32>());

        // batchd x sqed x headd
        let attn = (scores / (HEADD as f64).sqrt()).unwrap().matmul(&v).unwrap();

        Ok(self.proj.forward(&attn).unwrap())
    }
}

pub fn simple_llm(vs: VarBuilder, tkn: usize) -> impl Module{
    (0..3).map(|n| seq().add(DecoderBlock::new(vs.pp(format!("decoder_block_{n}")), DecoderConfig::default()).unwrap()).add(|xs: &Tensor| xs.relu()))
        .fold(seq(), |s, n| s.add(move |xs: &Tensor| (n.forward(xs) + xs).unwrap() * 0.5 ) )
        .add(linear(EMBD, tkn, vs.pp("output projection")).unwrap())
}

pub fn main() -> Result<()>{
    let batch_size = 40;

    let dev = Device::Cpu;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let decoder = simple_llm(vs.clone(), 255);

    let embeddings = Var::randn(0f32, 1f32, (255, EMBD), &dev)?;
    
    let mut vars = varmap.all_vars();
    vars.push(embeddings.clone());

    let mut opt = AdamW::new_lr(vars, 0.01)?;

    let mut file = BufReader::new(File::open("tiny-shakespeare.txt")?);
    let mut cost_sum = 0.;
    for n in 0..{
        let seqd = n % 80 + 2;

        let mut ys = vec![];
        let mut xs = vec![];

        for _n in 0..batch_size{
            let mut tokens = vec![0u8; seqd+1];
            file.read_exact(&mut tokens).expect("EOF");

            ys.push(Tensor::new(&tokens[1..seqd+1], &dev)?);
            let toks = Tensor::new(&tokens[0..seqd], &dev)?;

            xs.push(embeddings.as_tensor().embedding(&toks).unwrap().reshape((1, seqd, EMBD)).unwrap());
        }

        if seqd < 2{
            continue
        }

        let xs = Tensor::cat(&xs, 0).unwrap();
        //let ys = Tensor::cat(&ys, 0).unwrap();

        let yhat = decoder.forward(&xs).unwrap();

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
            for row in &yhat.to_vec3().unwrap()[0]{
                print!("{}", argmax(&row) as u8 as char)
            }
            println!();
        }
    }

    Ok(())
}
