#![feature(test)]

use std::{io::{BufReader, BufRead}, fs::File};
use ml_experiments::{*, rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel}};
use kdtree::{KdTree, distance::squared_euclidean};

pub const TOP_500: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top500.txt");
pub const TOP_250: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top250.txt");
pub const TOP_1000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top1000.txt");
pub const TOP_5000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top5000.txt");
pub const TOP_2500: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top2500.txt");
pub const TOP_10000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top10000.txt");
pub const TOP_37000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/top37000.txt");
pub const NOUNS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/nouns.txt");
pub const VERBS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/verbs.txt");

fn cart(a: &[f32], b: &[f32]) -> f32{
    a.iter().zip(b.iter()).map(|(a, b)| (a-b).powi(2) ).sum::<f32>().sqrt()
}

pub struct VecEmb{
    model: SentenceEmbeddingsModel,
    dict: Vec<(String, Vec<f32>)>
}

impl VecEmb{
    pub fn new(words: &str) -> Self{
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model().unwrap();

        let dict = BufReader::new(File::open(words).expect(words))
            .lines()
            .map(|word| {
                let word = word.unwrap();
                let [ref emb] = model.encode(&[word.clone()]).unwrap()[..] else {unreachable!()};
                (word, emb.clone())
            })
            .collect::<Vec<(String, Vec<f32>)>>();

        Self{model, dict}
    }

    pub fn tok(&self, word: &str) -> usize{
        let Self{dict, model} = self;
    
        let [ref emb1] = model.encode(&[word]).unwrap()[..] else {unreachable!()};
        let mut i = 0;
        let mut d1 = f32::INFINITY;

        for (j, (_, emb2)) in dict.iter().enumerate(){
            let d2 = cart(&emb1, &emb2);
            if d2 < d1{
                d1 = d2;
                i = j;
            }
        }

        i
    }

    pub fn str(&self, tok: usize) -> &str{
        &self.dict.get(tok).expect(&format!("No such token: {tok}")).0
    } 
}

pub struct HashEmb{
    model: SentenceEmbeddingsModel,
    kd: KdTree<f32, usize, Vec<f32>>,
    words: Vec<String>
}

impl HashEmb{
    pub fn new(word_dir: &str) -> Self{
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model().unwrap();
        
        let mut kd = KdTree::new(384);
        let mut words = vec![];

        for word in BufReader::new(File::open(word_dir).expect(word_dir)).lines(){
            let word = word.unwrap();
            let [ref emb] = model.encode(&[word.clone()]).unwrap()[..] else {unreachable!()};
            kd.add(emb.clone(), words.len()).expect(&format!("{}", emb.len()));
            words.push(word);
        }

        Self{model, kd, words}
    }

    pub fn tok(&self, word: &str) -> usize{
        let [ref emb1] = self.model.encode(&[word]).unwrap()[..] else {unreachable!()};
        *self.kd.nearest(emb1, 1, &squared_euclidean).unwrap()[0].1
    }

    pub fn str(&self, tok: usize) -> &str{
        self.words.get(tok).expect(&format!("No such token: {tok}"))
    }
}

pub type Embedder = HashEmb;

extern crate test;

#[bench]
fn hash_embedding(b: &mut test::Bencher) {
    let bruh = HashEmb::new(TOP_500);
    b.iter(|| bruh.tok("house"))
}

#[bench]
fn vec_embedding(b: &mut test::Bencher) {
    let bruh = VecEmb::new(TOP_500);
    b.iter(|| bruh.tok("house"))
}
