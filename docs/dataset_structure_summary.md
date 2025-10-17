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

## Log id mappings

- 1012 DeletePC
- 1013 PcLevelUp
- 1101 InviteParty
- 1102 JoinParty
- 1103 RefuseParty
- 1202 Die
- 1404 DuelEnd(PC)
- 1406 DuelEnd(Team)
- 1422 PartyBattleEnd(Team)
- 2112 ExpandWarehouse
- 2141 ChangeItemLook
- 2301 PutMainAuction
- 2405 UseGatheringItem
- 5004 CompleteQuest
- 5011 CompleteChallengeToday
- 5015 CompleteChallengeWeek
- 6001 CreateGuild
- 6002 DestoryGuild
- 6004 InviteGuild
- 6005 JoinGuild
- 6009 DissmissGuild
