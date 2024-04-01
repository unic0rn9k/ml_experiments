#![feature(test)]

use std::{io::{BufRead, BufReader}, fs::File, sync::{Mutex, Arc}, thread, collections::HashSet};
use ml_experiments::rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use clustering::*;

pub const TOP_37000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/top37000.txt");

fn main(){
    println!("1. Reading");
    let words: Arc<Vec<String>> = Arc::new(BufReader::new(File::open(TOP_37000).unwrap()).lines().map(|l| l.unwrap()).collect());
    
    println!("2. Embedding");
    let mut handles = vec![];
    let batch_size = 200;

    for i in 0..8{
        let words = words.clone();
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
                ret.push(model.encode(&words[n..n+batch_size]).unwrap())
            }
            ret
        }))
    }
    let mut handles: Vec<_> = handles.into_iter().map(|h| h.join().unwrap().into_iter()).collect();

    println!("2.5 Embedding synchronous");

    let mut embeddings: Vec<Vec<f32>> = vec![];
    let mut done: HashSet<usize> = (0..8).collect();
    loop{
        for n in done.clone().iter(){
            match handles[*n].next(){
                Some(mut some) => embeddings.append(&mut some),
                None => {done.remove(n);},
            }
        }
        if done.len() == 0{
            break
        }
    }

    println!("3. Clustering");
    
    let clustering = kmeans(800, &embeddings, 100);

    println!("membership: {:?}", clustering.membership);
    println!("centroids : {:?}", clustering.centroids);
}

mod kmeans_bench{
    use rand;
    use once_cell::sync::Lazy;
    extern crate test;
    use test::bench::*;

    static DATA: Lazy<Vec<Vec<f32>>> = Lazy::new(||{
        let mut samples: Vec<Vec<f32>> = vec![];
        for _ in 0..1000 {
            samples.push((0..384).map(|_| rand::random()).collect::<Vec<_>>());
        }
        samples
    });

    #[bench]
    fn clustering(b: &mut Bencher){
        use clustering::*;
        let samples = &*DATA;

        b.iter(|| kmeans(100, &samples, 200))
    }

    use linfa::DatasetBase;
    use linfa::traits::{Fit, FitWith, Predict};
    use linfa_clustering::{KMeansParams, KMeans, IncrKMeansError};
    use linfa_datasets::generate;
    use ndarray::{Axis, array, s};
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;
    use approx::assert_abs_diff_eq;

    #[bench]
    fn linfa(b: &mut Bencher){
        let observations = DatasetBase::from(DATA);
    }
}
