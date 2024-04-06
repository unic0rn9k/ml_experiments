#![feature(test)]

use std::{io::{BufRead, BufReader, Write}, fs::File, sync::Arc, thread, collections::HashSet, path::Path};
use ml_experiments::{rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType}, safetensors::{Dtype, tensor::TensorView}};
use kmedoids::arrayadapter::LowerTriangle;
use ml_experiments::{indicatif::{ProgressBar, ProgressState, ProgressStyle}, safetensors};

pub const TOP_37000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/top37000.txt");

fn dist(a: &[f32], b: &[f32]) -> f32{
    a.iter().zip(b.iter()).map(|(a, b)| (a-b).powi(2)).sum::<f32>().sqrt()
}

fn dissimilarity(m: Vec<&[f32]>) -> LowerTriangle<f32>{
    let rows = m.len();
    let cols = m[0].len();
    
    let mut data = vec![];

    for i in 0..rows{
        for j in (i+1)..rows{
            assert_eq!(m[i].len(), cols);
            assert_eq!(m[j].len(), cols);

            data.push(dist(&m[i], &m[j]))
        }
    }

    LowerTriangle { n: rows, data }
}

fn main(){
    println!("1.  Reading");
    let words: Arc<Vec<String>> = Arc::new(BufReader::new(File::open(TOP_37000).unwrap()).lines().map(|l| l.unwrap()).collect());
    
    println!("2.  Embedding");

    let batch_size = 200;
    let threads = 12;

    let pb = ProgressBar::new((words.len()/batch_size) as u64);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1} min", state.eta().as_secs_f64() as usize / 60).unwrap())
        .progress_chars("=> "));

    let mut handles = vec![];

    for i in 0..threads{
        let words = words.clone();
        let pb = pb.clone();
        handles.push(thread::spawn(move ||{

            let model = Arc::new(SentenceEmbeddingsBuilder::remote(
                SentenceEmbeddingsModelType::AllMiniLmL12V2
            ).create_model().unwrap());

            let mut ret = vec![];
            for j in 0..{
                let n = (j*8 + i) * batch_size;
                if n > words.len()-batch_size{
                    break
                }
                ret.push(model.encode(&words[n..n+batch_size]).unwrap());
                pb.inc(1);
            }
            ret
        }))
    }
    let mut handles: Vec<_> = handles.into_iter().map(|h| h.join().unwrap().into_iter()).collect();

    let mut embeddings: Vec<f32> = vec![];
    let mut done: HashSet<usize> = (0..threads).collect();
    let mut n_embeds = 0;
    loop{
        for n in done.clone().iter(){
            match handles[*n].next(){
                Some(some) => for mut some in some{
                    n_embeds += 1;
                    embeddings.append(&mut some)
                },
                None => {done.remove(n);},
            }
        }
        if done.len() == 0{
            break
        }
    }
    pb.finish();

    let embd = embeddings.len() / n_embeds;

    println!("... Saving");
    let num_bytes = std::mem::size_of::<f32>();
    let data = TensorView::new(Dtype::F32, vec![n_embeds, embd], unsafe{
        std::slice::from_raw_parts(embeddings.as_ptr() as *const u8, embeddings.len() / num_bytes)
    }).unwrap();
    safetensors::serialize_to_file(vec![("embeddings", data)], &None, Path::new("embeddings.safetensors")).unwrap();

    println!("4. Generating dissimilarity matrix");
    let dissim = dissimilarity((0..n_embeds).map(|n| &embeddings[n*embd..(n+1)*embd] ).collect());

    println!("3.  Clustering");
    let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
    let (loss, assingment, n_iter, n_swap): (f32, _, _, _) = kmedoids::fasterpam(&dissim, &mut meds, 100);
    
    println!("loss: {loss:?}");
    println!("assignment: {assingment:?}");
}
