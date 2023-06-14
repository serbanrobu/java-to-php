use clap::Parser;
use color_eyre::{
    eyre::{eyre, Context, ContextCompat},
    Result,
};
use ignore::WalkBuilder;
use indicatif::ProgressBar;
use reqwest::{
    header::{HeaderMap, AUTHORIZATION},
    Client,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tiktoken_rs::get_completion_max_tokens;
use tokio::task::JoinSet;

#[derive(Debug, Serialize)]
struct Request {
    model: &'static str,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Response {
    Ok { choices: Vec<Choice> },
    Err { error: Error },
}

#[derive(Debug, Deserialize)]
struct Error {
    message: String,
}

#[derive(Debug, Deserialize)]
struct Choice {
    text: String,
}

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    #[arg(short('k'), long, env("OPENAI_API_KEY"), help("OpenAI API key"))]
    api_key: String,
    #[arg(help("Source file or directory"))]
    source: PathBuf,
    #[arg(help("Destination directory"))]
    destination: PathBuf,
}

async fn convert(
    source_file_path: impl AsRef<Path>,
    destination_file_path: impl AsRef<Path>,
    client: &Client,
) -> Result<()> {
    let content = fs::read_to_string(source_file_path)?;
    let model = "text-davinci-003";
    let prompt = format!("#Java to PHP:\nJava:\n{}\n\nPHP:", content);
    let max_tokens = get_completion_max_tokens(model, &prompt).map_err(|e| eyre!(e))?;

    let request = Request {
        model,
        prompt,
        max_tokens,
        temperature: 0.,
    };

    let response = client
        .post("https://api.openai.com/v1/completions")
        .json(&request)
        .send()
        .await?
        .json::<Response>()
        .await?;

    let choices = match response {
        Response::Ok { choices } => choices,
        Response::Err { error } => return Err(eyre!("{}", error.message)),
    };

    let new_content = choices
        .first()
        .wrap_err("No choice received")?
        .text
        .as_str();

    fs::write(&destination_file_path, new_content)?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let Args {
        source,
        destination,
        api_key,
    } = Args::parse();

    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, format!("Bearer {}", api_key).parse()?);

    let client = Client::builder().default_headers(headers).build()?;

    if !destination.is_dir() {
        return Err(eyre!("{}: Not a directory", destination.display()));
    }

    if source.is_file() {
        let file_name = source.file_name().wrap_err("Invalid file name")?;
        let mut new_path = destination;
        new_path.push(file_name);
        new_path.set_extension("php");
        return convert(source, new_path, &client).await;
    }

    if !source.is_dir() {
        return Err(eyre!("{}: No such file or directory", source.display()));
    }

    let bar = ProgressBar::new(0);
    let mut tasks = JoinSet::<Result<()>>::new();

    for result in WalkBuilder::new(&source)
        .filter_entry(|entry| {
            let path = entry.path();

            path.is_dir()
                || path.is_file() && path.extension().map(|ext| ext == "java").unwrap_or(false)
        })
        .build()
    {
        let entry = result?;
        let path = entry.into_path();
        let relative_path = path.strip_prefix(&source)?;
        let mut new_path = destination.clone();
        new_path.push(relative_path);

        if path.is_file() {
            new_path.set_extension("php");
            let client = client.clone();

            tasks.spawn(async move {
                convert(path, &new_path, &client)
                    .await
                    .wrap_err_with(|| eyre!("{}", new_path.display()))
            });
        } else if !new_path.exists() {
            fs::create_dir(&new_path)?;
        }
    }

    bar.set_length(tasks.len() as u64);

    while let Some(result) = tasks.join_next().await.transpose()? {
        if let Err(e) = result {
            bar.println(format!("{:#}", e));
        }

        bar.inc(1);
    }

    bar.finish();

    Ok(())
}
