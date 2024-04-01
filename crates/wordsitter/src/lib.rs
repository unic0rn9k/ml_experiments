use std::collections::HashMap;
use ml_experiments::{*, rust_bert::pipelines::pos_tagging::POSTag};


// 1. **NN** (Noun, singular): Represents a single noun, such as "apple", "table", or "chair".
// 2. **NNS** (Noun, plural): Represents a plural noun, such as "apples", "tables", or "chairs".
// 3. **NNP** (Proper Noun, singular): Represents the name of a specific person, place, organization, etc., such as "John", "Paris", or "Microsoft".
// 4. **NNPS** (Proper Noun, plural): Represents a plural proper noun, such as "Smiths" or "Joneses".
// 5. **DT** (Determiner): Identifies words that help to describe or quantify a noun or pronoun, such as "the", "a", "an", "every", or "some".
// 6. **PRP** (Personal Pronoun): Represents a pronoun referring to a person, such as "I", "you", "he", "she", or "they".
// 7. **PRDT** (Determiner, possessive): Represents a determiner that shows possession, such as "my", "your", "his", "her", or "their".
// 8. **VBG** (Verb, gerund/present participle): Represents a verb in its present participle form, which functions as an adjective to modify a noun, such as "swimming" or "running".
// 9. **VBN** (Verb, past tense/participle): Represents a verb in its past tense or past participle form, such as "ate", "eaten", or "drank".
// 10. **VBZ** (Verb, third person singular present): Represents a verb in its base form that describes an action happening to a third-person singular subject, such as "loves" or "swims".
// 11. **VBD** (Verb, past tense/other non-third person singular): Represents a verb in its past tense form that does not refer to a third-person singular subject, such as "had loved" or "have loved".
// 12. **JJ** (Adjective): Represents an adjective, which modifies a noun or pronoun and describes its qualities or characteristics, such as "red", "big", or "happy".
// 13. **JJR** (Adjective, comparative): Represents a comparative adjective, which compares the degree of quality between two things, such as "more beautiful" or "taller than".
// 14. **JJS** (Adjective, superlative): Represents a superlative adjective, which modifies a noun and describes the highest degree of a quality among multiple things, such as "the most beautiful" or "the happiest".
// 15. **RB** (Adverb, general): Represents an adverb, which modifies a verb, adjective, or another adverb, and provides more information about how an action is performed or the degree to which it is done, such as
// "quickly" or "loudly".
// 16. **RBR** (Adverb, comparative): Represents a comparative adverb, which compares the degree of something between two things, such as "more quickly" or "sooner than".
// 17. **RBS** (Adverb, superlative): Represents a superlative adverb, which modifies an adjective and describes the highest degree of comparison, such as "most quickly" or "barely at all".
// 18. **CC** (Coordinating conjunction): Represents a coordinating conjunction, which connects two independent clauses, such as "and", "but", or "or".
// 19. **IN** (Subordinating conjunction/preposition/other): Represents a subordinating conjunction, preposition, or other type of word that introduces a dependent clause or modifies a noun phrase, such as "because",
// "after", or "in the kitchen".
// 20. **TO** (Preposition/infinitive marker): Represents a preposition or infinitive marker, which introduces a prepositional phrase or shows the infinitive form of a verb, such as "to" in "I want to go home" or
// "into" in "He ran into the room".
// 21. **WRB** (Wh-word): Represents a wh-word (interrogative pronoun), which introduces a question by asking for information about the subject, verb, or object, such as "what", "where", or "why".
// 22. **WDT** (Wh-determiner): Represents a wh-determiner, which modifies a wh-word to introduce a relative clause that defines or describes an antecedent, such as "which" or "that".
// 23. **WDJ** (Wh-adverb): Represents a wh-adverb, which modifies a wh-word and asks for more information about the time, place, manner, or reason of an action, such as "when", "where", or "how".
// 24. **CCG** (Coordinating conjunction, subordinating): Represents a coordinating conjunction that connects two coordinate clauses with the same grammatical function, such as "although" or "while".
// 25. **CCONJ** (Conjunction, other): Represents any other type of conjunction, such as "nor", "but", or "yet", which connects words, phrases, or clauses in various ways.


#[derive(Debug, Clone, Copy)]
pub enum Class{
    Noun,
    Verb,
    Unk
}

fn onehot(n: usize, l: usize) -> Vec<f32>{
    let mut ret = vec![0f32; l];
    for (i, j) in ret.iter_mut().enumerate(){
        if i == n{
            *j=1.
        }
    }
    ret
}

impl Class{
    fn onehot(self) -> Vec<f32>{
        onehot(self as usize, 3)
    }
}

pub struct NodeId{
    class: Class,
    id: usize,
}

pub struct Sitted{
    max_id: usize,
    identifiers: HashMap<NodeId, String>,
    tokens: Vec<NodeId>
}

impl Sitted{
    pub fn embed(&self) -> Vec<Vec<f32>>{
        let mut emb = vec![];
        for tok in &self.tokens{
            let mut ret = tok.class.onehot();
            ret.append(&mut &mut onehot(tok.id, self.max_id));
            emb.push(ret)
        }
        emb
    }
}

#[test]
fn pos(){
    use rust_bert::pipelines::pos_tagging::POSModel;
    let pos_model = POSModel::new(Default::default()).unwrap();
    let input = ["The idea of this exercise is to create a minimal subset language of English. Next I will need a similar, or larger sized list. Here it is important that the words can convey a large variation of \"props\" (weapon, blade, cicle) or characters (dragon, hero, trader, merchant), phenomenons (eg. illness, party), concepts (eg. analogy, reference, concept, process), emotions (happiness, sorrow) and generic objects that are common in the physical world (eg. plant, animal, table, vehicle). Feel free to use my examples."];
    let output = pos_model.predict(&input);
    for word in output[0].iter(){
        print!("{}:{} ", word.word, word.label)
    }
    println!()
}

pub fn tokenize(text: &str) -> Sitted{
    todo!()
}
