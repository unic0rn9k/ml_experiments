use std::{collections::HashSet, fs::File};
use std::io::{BufRead, BufReader};

use top_word_embedding::*;

const SAMPLE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/sample.txt"));

//fn main(){
//    let emb = Embedder::new(TOP_37000);
//
//    for line in SAMPLE.lines(){
//        for word in line.split(" "){
//            print!("{} ", emb.str(emb.tok(word)));
//        }
//        println!()
//    }
//}

fn main(){
    let file = BufReader::new(File::open(TOP_37000).unwrap()).lines();
    let set: HashSet<String> = file.map(|v|v.unwrap()).collect();
    let valid_chars: HashSet<char> =  ('a'..='z').chain([' ']).collect();
    let s = |s: &str| -> String { s.chars().filter(|c| valid_chars.contains(c)).collect() };

    println!("{}", SAMPLE.to_ascii_lowercase());

    for word in SAMPLE.to_ascii_lowercase().split(" ").map(s){
        if !set.contains(&word){
            println!("unk: {word:?}")
        }
    }
}
