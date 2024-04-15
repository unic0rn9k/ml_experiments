//#![feature(test)]
//
//use std::{io::{BufRead, BufReader, Write}, fs::File, sync::Arc, thread, collections::HashSet, path::Path, time::SystemTime};
//use ml_experiments::{rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType}, safetensors::{Dtype, tensor::TensorView, SafeTensors}};
//use kmedoids::arrayadapter::LowerTriangle;
//use ml_experiments::{memmap2::MmapOptions, indicatif::{ProgressBar, ProgressState, ProgressStyle}, safetensors};
//
//pub const TOP_37000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/top37000.txt");
//
//fn dist(a: &[f32], b: &[f32]) -> f32{
//    a.iter().zip(b.iter()).map(|(a, b)| (a-b).powi(2)).sum::<f32>().sqrt()
//}
//
//fn dissimilarity(m: Vec<&[f32]>) -> LowerTriangle<f32>{
//    let rows = m.len();
//    let cols = m[0].len();
//    
//    let mut data = vec![];
//
//    for i in 0..rows{
//        for j in (i+1)..rows{
//            assert_eq!(m[i].len(), cols);
//            assert_eq!(m[j].len(), cols);
//
//            data.push(dist(&m[i], &m[j]))
//        }
//    }
//
//    LowerTriangle { n: rows, data }
//}
//
//fn embeddings(words: Arc<Vec<String>>) -> Vec<f32>{
//    println!("2.  Embedding");
//
//    let batch_size = 200;
//    let threads = 2;
//
//    let pb = ProgressBar::new((words.len()/batch_size) as u64);
//    pb.set_style(ProgressStyle::with_template(" Embedding [{bar:.green}] ({eta})")
//        .unwrap()
//        .with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1} min", state.eta().as_secs_f64() as usize / 60).unwrap())
//        .progress_chars("=> "));
//
//    let mut handles = vec![];
//
//    for i in 0..threads{
//        let words = words.clone();
//        let pb = pb.clone();
//        handles.push(thread::spawn(move ||{
//
//            let model = Arc::new(SentenceEmbeddingsBuilder::remote(
//                SentenceEmbeddingsModelType::AllMiniLmL12V2
//            ).create_model().unwrap());
//
//            let mut ret = vec![];
//            for j in 0..{
//                let n = (j*8 + i) * batch_size;
//                if n > words.len()-batch_size{
//                    break
//                }
//                ret.push(model.encode(&words[n..n+batch_size]).unwrap());
//                pb.inc(1);
//            }
//            ret
//        }))
//    }
//    let mut handles: Vec<_> = handles.into_iter().map(|h| h.join().unwrap().into_iter()).collect();
//
//    let mut embeddings: Vec<f32> = vec![];
//    let mut done: HashSet<usize> = (0..threads).collect();
//    loop{
//        for n in done.clone().iter(){
//            match handles[*n].next(){
//                Some(some) => for mut some in some{
//                    embeddings.append(&mut some)
//                },
//                None => {done.remove(n);},
//            }
//        }
//        if done.len() == 0{
//            break
//        }
//    }
//
//    let embd = 384;
//    let n_embeds = embeddings.len()/embd;
//
//    println!("... Saving");
//    let num_bytes = std::mem::size_of::<f32>();
//    let data = TensorView::new(Dtype::F32, vec![n_embeds, embd], unsafe{
//        std::slice::from_raw_parts(embeddings.as_ptr() as *const u8, embeddings.len() * num_bytes)
//    }).unwrap();
//    safetensors::serialize_to_file(vec![("embeddings", data)], &None, Path::new("embeddings.safetensors")).unwrap();
//
//    pb.finish();
//
//    embeddings
//}
//
//fn time_it<T>(f: impl Fn()->T) -> T{
//    let a = SystemTime::now();
//    let ret = f();
//    let b = SystemTime::now();
//    let dur = b.duration_since(a).unwrap();
//    println!("Took {dur:?}");
//    ret
//}
//
//fn main(){
//    println!("1.  Reading");
//    let words: Arc<Vec<String>> = Arc::new(BufReader::new(File::open(TOP_37000).unwrap()).lines().map(|l| l.unwrap()).collect());
//    
//    let embeddings = if Path::new("embeddings.safetensors").exists(){
//        // TODO: Load embeddings
//        todo!()
//        //let file = File::open("embeddings.safetensors").unwrap();
//        //let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
//        //let tensors = SafeTensors::deserialize(&buffer).unwrap();
//        //let tensor = tensors
//        //    .tensor("embeddings")
//        //    .unwrap();
//        //tensor.data()
//    }else{
//        time_it(||embeddings(words.clone()))
//    };
//
//    let embd = 384;
//    let n_embeds = embeddings.len()/embd;
//
//    println!("4. Generating dissimilarity matrix");
//    let dissim = dissimilarity((0..n_embeds).map(|n| &embeddings[n*embd..(n+1)*embd] ).collect());
//
//    // TODO: Load clusterings 
//    println!("3.  Clustering");
//    let mut meds = kmedoids::random_initialization(n_embeds, 800, &mut rand::thread_rng());
//    let (loss, assignment, n_iter, n_swap): (f32, _, _, _) = kmedoids::fasterpam(&dissim, &mut meds, 100);
//    
//    println!("loss: {loss:?}");
//
//    let mut out = File::create("assignments.txt").unwrap();
//    for (i, n) in assignment.iter().enumerate(){
//        let suffix = if i == n_embeds{","}else{""};
//        write!(out, "{n}{suffix}").unwrap();
//    }
//}

use ml_experiments::decoder::simple_llm;
use morphological_tokenization::*;
use ml_experiments::*;
use std::io::Read;

use std::iter::repeat_with;
use std::{fs::File, io::BufRead};
use std::io::BufReader;
use std::ops::Deref;

use candle_nn::encoding::one_hot;
use candle_nn::ops::log_softmax;
use candle_nn::{*, ops::softmax};
use candle_core::*;

const EMBD: usize = 100;
const HEADD: usize = 300;
const NHEADS: usize = 4;

pub fn main() -> Result<()>{
    let dev = Device::Cpu;

    let batch_size = 40;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let decoder = simple_llm(vs.clone(), *TOKENN as usize);

    let embeddings = Var::randn(0f32, 1f32, (*TOKENN as usize, EMBD), &dev)?;
    
    let mut vars = varmap.all_vars();
    vars.push(embeddings.clone());

    let mut opt = AdamW::new_lr(vars, 0.01)?;

    let mut file = File::open("bruh.txt")?;
    let mut cost_sum = 0.;
    for n in 0..{
        let seqd = (n*2) % 40 + 2;

        let mut ys = vec![];
        let mut xs = vec![];

        for _n in 0..batch_size{
            let tokens: Vec<u32> = TokenizerStream::new(&mut file).take(seqd+2).collect();

            ys.push(Tensor::new(&tokens[2..seqd+2], &dev)?);
            let toks = Tensor::new(&tokens[0..seqd], &dev)?;

            xs.push(embeddings.as_tensor().embedding(&toks).unwrap().reshape((1, seqd, EMBD)).unwrap());
        }

        if seqd < 2{
            unreachable!()
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
            let mut buffer = vec![];
            for row in &yhat.to_vec3().unwrap()[0]{
                buffer.push(argmax(&row) as u32)
            }
            println!("{}", decode(&buffer));
        }
    }

    Ok(())
}
