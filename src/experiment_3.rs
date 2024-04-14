use crate::*;
use std::io::Read;

pub fn main() -> Result<()>{
    let batch_size = 40;

    let dev = Device::Cpu;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let decoder = simple_llm(vs.clone());

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
            cost_sum = 0.;
        }
    }

    Ok(())
}
