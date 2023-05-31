use clap::Parser;
use color_eyre::{
    eyre::{eyre, ContextCompat},
    Result,
};
use ignore::WalkBuilder;
use indicatif::ProgressBar;
use reqwest::{
    header::{self, AUTHORIZATION},
    Client,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
struct Request {
    model: &'static str,
    messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    Assistant,
    System,
    User,
}

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: Role,
    content: String,
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

    let request = Request {
        model: "gpt-3.5-turbo",
        messages: vec![
            Message {
                role: Role::System,
                content: "You are a java to php converter".into(),
            },
            Message {
                role: Role::User,
                content,
            },
        ],
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .json(&request)
        .send()
        .await?
        .json::<Response>()
        .await?;

    let content = &response
        .choices
        .first()
        .wrap_err("No choice received")?
        .message
        .content;

    fs::write(&destination_file_path, content)?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args {
        source,
        destination,
        api_key,
    } = Args::parse();

    let mut headers = header::HeaderMap::new();
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
    let mut handles = vec![];

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
            let bar = bar.clone();
            let client = client.clone();

            let handle = tokio::spawn(async move {
                convert(path, new_path, &client).await?;

                bar.inc(1);

                Ok(()) as Result<_>
            });

            handles.push(handle);
        } else if !new_path.exists() {
            fs::create_dir(&new_path)?;
        }
    }

    bar.set_length(handles.len() as _);

    for handle in handles {
        if let Err(e) = handle.await? {
            bar.println(e.to_string());
        }
    }

    bar.finish();

    Ok(())
}
