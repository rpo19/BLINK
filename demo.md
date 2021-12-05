# BLINK api demo
The [setup](setup.md) is required.

## Configure the demo
The demo configurations are customizable by modifying the main file `blink/main_api.py`.

## Run the demo
```
python blink/main_api.py [host] [port]
```
The wait for all the models to load in memory (it generally takes some minutes).

## APIs
Currently two APIs are available:
```
/api/entity-link/text
# Starting from text it runs NER using flair and then it runs BLINK NEL.
```
```
/api/entity-link/samples
# Given a mention with its surface form, context left and context right it runs BLINK NEL.
```

## Example Calls
```
POST http://localhost:30300/api/entity-link/text
Host: localhost
Content-Type: application/json

{
    "text": "I just saw valentino rossi on a motorbike"
}
```
```
POST http://localhost:30300/api/entity-link/samples
Host: localhost
Content-Type: application/json

[
    {
        "label": "unknown",
        "label_id": -1,
        "context_left": "I just saw",
        "context_right": "on a motorbike",
        "mention": "valentino rossi",
        "start_pos": 10,
        "end_pos": 25,
        "sent_idx": 0
    },
    {
        "label": "unknown",
        "label_id": -1,
        "context_left": "Tomorrow I'll be in",
        "context_right": "",
        "mention": "Milan",
        "start_pos": 10,
        "end_pos": 25,
        "sent_idx": 0
    }
]
```
### cURL Commands
```
curl --request POST \
  --url http://localhost:30300/api/entity-link/text \
  --header 'content-type: application/json' \
  --header 'host: localhost' \
  --header 'user-agent: vscode-restclient' \
  --data '{"text": "I just saw valentino rossi on a motorbike"}'
```
```
curl --request POST \
  --url http://localhost:30300/api/entity-link/samples \
  --header 'content-type: application/json' \
  --header 'host: localhost' \
  --header 'user-agent: vscode-restclient' \
  --data '[{"label": "unknown","label_id": -1,"context_left": "I just saw","context_right": "on a motorbike","mention": "valentino rossi","start_pos": 10,"end_pos": 25,"sent_idx": 0},{"label": "unknown","label_id": -1,"context_left": "Tomorrow I'\''ll be in","context_right": "","mention": "Milan","start_pos": 10,"end_pos": 25,"sent_idx": 0}]'
```

## Example Response
```
[
  {
    "idx": idx,
    "sample": {...},
    "entity_id": id,
    "entity_title": title,
    "entity_text": text_description,
    "score": linking_score,
    "url": url,
    "crossencoder": true,
    "_nil_p": 0.9,
    "candidates": [
      {
        "entity": {
          "e_id": id,
          "e_title": title,
          "e_url": url,
          "e_text": text_description
        },
        "score": score
      },
      ...
    ]
  },
  ...
]
```
The `_nil_p` value is the estimated probability of the best candidate to be the correct one; in other words values close to `1` mean the linking is likely correct while values close to `0` mean the mention is probably `NIL`.