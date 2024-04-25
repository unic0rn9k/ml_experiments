use std::{io::Write, thread, sync::{Arc, RwLock}};

use ml_experiments::{rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel, indicatif::{ProgressBar, ProgressStyle, ProgressState}};
use rand::{thread_rng, Rng};

use crate::*;

pub fn main(){
    let idxs: Vec<usize> = repeat_with(|| thread_rng().gen_range(0..WORDS.len())).take(500).collect();
    let words: Arc<Vec<&str>> = Arc::new(idxs.into_iter().map(|n|WORDS[n].as_str()).collect());
    let candidate_labels = &["emotionally positive", "logical", "a grammer word", "emotionally negative", "commanding", "polite", "scientific", "complicated", "simple", "dumb", "gendered", "an inanimate object"];

    let threads = 1;
    let batch_size = words.len() / threads;
    let remainder = words.len() % threads;
    let mut handles = vec![];

    let pb = ProgressBar::new(words.len() as u64);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1} min", state.eta().as_secs_f64() as usize / 60).unwrap())
        .progress_chars("=> "));

    pb.tick();

    for n in 0..threads{
        let a = n * batch_size;
        let mut b = (n+1) * batch_size;
        let words = words.clone();
        let pb = pb.clone();

        handles.push(thread::spawn(move || {
            if n == threads-1{b += remainder}

            let model = ZeroShotClassificationModel::new(Default::default()).unwrap();

            pb.println(format!("Processing {a}..{b}/{} on thread {n}", words.len()));

            let classification = model.predict_multilabel(
                &words[a..b],
                candidate_labels,
                //Some(Box::new(|s|format!("The sample is {s}."))),
                None,
                128,
            ).unwrap();

            pb.println(format!("{}: {:.2} {}", words[0], classification[0][0].score, classification[0][0].text));

            classification
        }));
    }

    let mut i = 0;
    let mut file = File::create("classifications.txt").unwrap();
    for classification in handles.into_iter().map(|h|h.join().unwrap()).flatten(){

        let mut ret = 0u32;
        
        pb.println(format!("{}: {:.2} {}", words[i], classification[0].score, classification[0].text));
        i += 1;

        for class in classification.into_iter(){
            if class.score >= 0.5{
                ret |= 1<<class.id;
            }
        }

        writeln!(&mut file, "{ret}").unwrap();
        pb.inc(1);
    }
    pb.finish();
}
