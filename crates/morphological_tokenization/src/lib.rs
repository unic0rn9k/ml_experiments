#![feature(iter_array_chunks)]
use std::{io::{BufReader, BufRead, Read, Bytes}, fs::File, collections::HashMap, iter::Chain, str::Chars, array::IntoIter};
use once_cell::sync::Lazy;

pub const TOP_37000: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/words/top37000.txt");

/// Initial whitespace corresponds to the UNK token
pub const SPECIAL_TOKENS: &[&str] = &[" ", ",", "."];

pub static WORDS: Lazy<Vec<String>> = Lazy::new(||BufReader::new(File::open(TOP_37000).unwrap()).lines().map(|l| l.unwrap()).collect());
pub static WORD_IDX: Lazy<HashMap<String, u32>> = Lazy::new(||SPECIAL_TOKENS.iter().map(|s|s.to_string()).chain(WORDS.iter().cloned()).enumerate().map(|(a,b)| (b,a as u32)).collect());
pub static TOKENN: Lazy<u32> = Lazy::new(||(WORDS.len() as f32 + SPECIAL_TOKENS.len() as f32 + 2.).sqrt() as u32 + 1);

pub struct TokenizerStream<R: Iterator<Item=char>>(R, Vec<u32>);

impl<R: Iterator<Item=char>> TokenizerStream<R>{
    pub fn new(text: R) -> Self{
        Self(text, vec![])
    }
}

impl<R: Iterator<Item=char>> Iterator for TokenizerStream<R>{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = &mut self.1;
        let mut buffer = "".to_string();

        loop{
            if ret.len() != 0{
                return Some(ret.remove(0));
            }

            let unk = |ret: &mut Vec<u32>|
            if ret.len() == 0 || ret[ret.len()-1] != 0 as u32 || ret[ret.len()-2] != 0{
                //ret.push(0);
                //ret.push(0);
            };

            let tok = |ret: &mut Vec<u32>, buffer: &String|{
                if buffer.is_empty(){
                    return
                }

                if let Some(n) = WORD_IDX.get(buffer){
                    if *n >= *TOKENN**TOKENN - SPECIAL_TOKENS.len() as u32{
                        panic!("Token too large???")
                    }
                    ret.push(n / *TOKENN);
                    ret.push(n % *TOKENN);
                }else{
                    unk(ret);
                }
            };

            let c = self.0.next()?.to_ascii_lowercase() as char;

            if c.is_whitespace(){
                tok(ret, &buffer);
                buffer="".into();
                continue
            }

            if SPECIAL_TOKENS.iter().any(|n| n.chars().next().unwrap()==c){
                tok(ret, &buffer);
                tok(ret, &c.to_string());
                buffer="".into();
                continue
            }

            if !c.is_ascii_alphabetic(){
                tok(ret, &buffer);
                unk(ret);
                buffer="".into();
                continue;
            }

            buffer.push(c);
        }
    }
}

pub fn decode(toks: &[u32]) -> String{
    let mut buffer = "".to_string();

    for [a, b] in toks.iter().array_chunks(){
        if *a == 0 && *b == 0{
            buffer += " <UNK>";
            continue
        }

        let n = *a * *TOKENN + *b;

        if n >= SPECIAL_TOKENS.len() as u32{
            buffer += " ";
            buffer += &WORDS[n as usize - SPECIAL_TOKENS.len()];
        }else{
            buffer += SPECIAL_TOKENS[n as usize]
        }

    }

    buffer
}


#[test]
fn the_quick_brown_fox(){
    let text = File::open("/home/unic0rn9k/Documents/ml_experiments/bruh.txt").unwrap().bytes().map(|c| c.unwrap() as char).take(50000);

    let toks: Vec<_> = TokenizerStream::new(text).collect();
    let dec = decode(&toks);
    //println!("{text:?}");
    //println!();
    //println!("{}", *TOKENN);
    //println!();
    //println!("{toks:?} {}", toks.len());
    //println!();
    println!("{dec:?}");
    println!();
}
