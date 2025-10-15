- Each CSV in `raw` contains logs for a single player account
  - Each player account can have multiple characters
- `train_labeld.csv` contains churn labels for each account ID

## Player account logs

- `session` shows the ID of a session
  - If 0 => session stopped (visisble in the subsequent timestamps)
- `logid` shows what category this log belongs to
  - Has corresponding `log_detail_code` to get further information
  - Example:
    - Log ID 1005 is for `EnterZoneReasonCode`, when an actor enters a zone
    - But which actor? And why?
    - `log_detail_code` of 1 would mean player enters the game server (see `BnS_logSchema_Code.csv`)
    - And 3 would mean player got revided

## Preprocessing approach

- Read raw CSV logs via pandas
- Filter for necessary `logid` and `log_detail_code`
- Remove unnecessary columns
