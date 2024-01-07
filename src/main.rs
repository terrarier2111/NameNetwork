#![feature(string_remove_matches)]

mod utils;
mod network;

// data src: https://www.ssa.gov/oact/babynames/limits.html

use std::{fs::{self, read_dir, File, OpenOptions, create_dir}, path::Path, io::{Read, Write, Seek}};

use rand::{Rng, seq::SliceRandom};
use serde_derive::{Serialize, Deserialize};
use utils::count_occourances;

use crate::network::{NetworkBuilder, LayerBuilder};

fn main() {
    print_metadata();

    // this should by far be enough space for all first names we could ever need (famous last words)
    const INPUT_CHARS: usize = 32;
    const INPUT_SIZE: usize = INPUT_CHARS + 1 + 1 + 1; // for all 4 parameters (name(32), gender(1), popularity%(1), year(1))

    const OUTPUT_SIZE: usize = 2 + 1 + 1; // M/F, human (text) but no name, not human (text)

    let mode = Mode::Training;

    let mut rng = rand::thread_rng();
    let entries = if let Some(mut entries) = read_cache(mode) {
        entries.shuffle(&mut rng);
        entries
    } else {
        const DEV_DATA_PERCENTAGE: usize = 1;

        let mut training = read_uncached().unwrap();
        training.shuffle(&mut rng);
        let dev_entries = (training.len() / 100 + 1) * 1;
        let mut dev = Vec::with_capacity(dev_entries);
        for i in 0..dev_entries {
            dev.push(training.remove(dev_entries - 1 - i));
        }
        write_cache(&dev, &training);
        if mode == Mode::Training {
            training
        } else {
            dev
        }
    };

    let mut network = NetworkBuilder::new().learning_rate(0.05)
    .input_size(INPUT_SIZE).output_size(OUTPUT_SIZE).hidden(LayerBuilder::new().neurons(10)).build();
    // network.eval();
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Training,
    Dev,
}

#[derive(Serialize, Deserialize)]
#[repr(u8)]
enum Gender {
    Male = 0,
    Female = 1,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    name: String,
    year: usize,
    gender: Gender,
    popularity: usize,
}






fn read_uncached() -> Option<Vec<Entry>> {
    if !Path::new(DIR).exists() {
        return None;
    }
    let mut entries = vec![];
    for entry in read_dir(DIR).unwrap() {
        if let Ok(entry) = entry {
            if entry.file_type().unwrap().is_file() && entry.path().file_name().unwrap().to_str().unwrap().split_once('.').unwrap().1.eq_ignore_ascii_case("txt") {
                let mut file = OpenOptions::new().read(true).write(true).open(entry.path()).unwrap();
                let mut content = String::new();
                file.read_to_string(&mut content).unwrap();
                content.remove_matches('\r');
                for line in content.split('\n') {
                    if line.is_empty() {
                        continue;
                    }
                    if count_occourances(line, ',') != 2 {
                        println!("WARNING: Found invalid line \"{}\" in file \"{}\"", line, entry.path().to_str().unwrap());
                        continue;
                    }
                    let mut parts = line.split(',');
                    let year = entry.file_name().to_str().unwrap().split_once('.').unwrap().0.split_at(3).1.parse::<usize>().unwrap();
                    let name = parts.next().unwrap();
                    let gender = if parts.next().unwrap() == "M" {
                        Gender::Male
                    } else {
                        Gender::Female
                    };
                    let popularity = parts.next().unwrap().parse::<usize>().unwrap();
                    entries.push(Entry {
                        name: name.to_string(),
                        year,
                        gender,
                        popularity,
                    });
                }
            }
        }
    }
    Some(entries)
}

const DIR: &str = "./names/";
const CACHE_DIR: &str = "./cache/";
const TRAINING_FILE: &str = "./cache/training.json";
const DEV_FILE: &str = "./cache/dev.json";

fn write_cache(dev: &Vec<Entry>, training: &Vec<Entry>) {
    if !Path::new(CACHE_DIR).exists() {
        create_dir(CACHE_DIR).unwrap();
    }
    let mut training_file = File::create_new(TRAINING_FILE).unwrap();
    training_file.write_all(serde_json::to_string(training).unwrap().as_bytes()).unwrap();
    let mut dev_file = File::create_new(DEV_FILE).unwrap();
    dev_file.write_all(serde_json::to_string(dev).unwrap().as_bytes()).unwrap();
}

fn read_cache(mode: Mode) -> Option<Vec<Entry>> {
    if !Path::new(CACHE_DIR).exists() || !Path::new(TRAINING_FILE).exists() || !Path::new(DEV_FILE).exists() {
        return None;
    }
    match mode {
        Mode::Training => {
            let mut training_file = File::open(TRAINING_FILE).unwrap();
            let mut raw_training = String::new();
            training_file.read_to_string(&mut raw_training).unwrap();
            let training_entries: Vec<Entry> = serde_json::from_slice(raw_training.as_bytes()).unwrap();
            Some(training_entries)
        },
        Mode::Dev => {
            let mut dev_file = File::open(DEV_FILE).unwrap();
            let mut raw_dev = String::new();
            dev_file.read_to_string(&mut raw_dev).unwrap();
            let dev_entries: Vec<Entry> = serde_json::from_slice(raw_dev.as_bytes()).unwrap();
            Some(dev_entries)
        },
    }
}































fn print_metadata() {
    let dir = "./names/";
    if !Path::new(&dir).exists() {
        println!("FATAL: The data to be traversed couldn't be found!");
        return;
    }
    let mut longest_name = 0;
    for entry in read_dir(dir).unwrap() {
        if let Ok(entry) = entry {
            if entry.file_type().unwrap().is_file() && entry.path().file_name().unwrap().to_str().unwrap().split_once('.').unwrap().1.eq_ignore_ascii_case("txt") {
                let mut file = OpenOptions::new().read(true).write(true).open(entry.path()).unwrap();
                let mut content = String::new();
                file.read_to_string(&mut content).unwrap();
                content.remove_matches('\r');
                for line in content.split('\n') {
                    if line.is_empty() {
                        continue;
                    }
                    if count_occourances(line, ',') != 2 {
                        println!("WARNING: Found invalid line \"{}\" in file \"{}\"", line, entry.path().to_str().unwrap());
                        continue;
                    }
                    let parts = line.split_once(',').unwrap();
                    if parts.0.len() > longest_name {
                        longest_name = parts.0.len();
                    }
                }
            }
        }
    }
    println!("Longest name: {longest_name}");
}

fn remove_cnt_data() {
    let dir = "./names/";
    if !Path::new(&dir).exists() {
        println!("FATAL: The data to be curated couldn't be found!");
        return;
    }
    for entry in read_dir(dir).unwrap() {
        if let Ok(entry) = entry {
            if entry.file_type().unwrap().is_file() && entry.path().file_name().unwrap().to_str().unwrap().split_once('.').unwrap().1.eq_ignore_ascii_case("txt") {
                println!("Modifying {}", entry.path().to_str().unwrap());
                let mut file = OpenOptions::new().read(true).write(true).open(entry.path()).unwrap();
                let mut content = String::new();
                file.read_to_string(&mut content).unwrap();
                let mut result = String::with_capacity(content.len());
                content.remove_matches('\r');
                let mut first = true;
                for line in content.split('\n') {
                    if line.is_empty() {
                        continue;
                    }
                    if count_occourances(line, ',') != 2 {
                        println!("WARNING: Found invalid line \"{}\" in file \"{}\"", line, entry.path().to_str().unwrap());
                        continue;
                    }
                    if !first {
                        // FIXME: handle errors gracefully!
                        result.push('\n');
                    }
                    first = false;

                    let parts = line.split(',');

                    let mut invalid = false;
                    let mut pushed = false;
                    'part: for (i, part) in parts.enumerate() {
                        let mut num = false;
                        
                        for chr in part.chars().into_iter() {
                            if chr.is_ascii_alphanumeric() && !chr.is_ascii_alphabetic() {
                                num = true;
                                if i != 2 {
                                    invalid = true;
                                    break 'part;
                                }
                                break;
                            }
                            // TODO: is this condition sufficient?
                            if !chr.is_ascii_alphabetic() {
                                invalid = true;
                                break;
                            }
                        }
                        if num {
                            continue;
                        }
                        if pushed {
                            // FIXME: handle errors gracefully!
                            result.push(',');
                        }
                        pushed = true;
                        // FIXME: handle errors gracefully!
                        result.push_str(part);
                    }
                    if invalid {
                        println!("WARNING: Found invalid line \"{}\" in file \"{}\"", line, entry.path().to_str().unwrap());
                        continue;
                    }
                }
                file.seek(std::io::SeekFrom::Start(0)).unwrap();
                file.set_len(0).unwrap();
                file.write_all(result.as_bytes()).unwrap();
            }
        }
    }
}
