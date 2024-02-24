use top_word_embedding::*;

const SAMPLE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/sample.txt"));

fn main(){
    let emb = Embedder::new(TOP_250);

    for line in SAMPLE.lines(){
        for word in line.split(" "){
            print!("{} ", emb.str(emb.tok(word)));
        }
        println!()
    }
}
