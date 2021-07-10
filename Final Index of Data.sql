SELECT BI.Player_Name, BI.Seasons, BI.Team, PD.Position, 
	MAX(CASE 
		WHEN AD.Goals_with_Left_Foot > AD.Goals_with_Right_Foot THEN 'Lefty' 
		WHEN AD.Goals_with_Left_Foot < AD.Goals_with_Right_Foot THEN 'Righty'
		WHEN AD.Goals_with_Left_Foot = AD.Goals_with_Right_Foot THEN 'Ambidextrous'
		ELSE 'None' END)
		OVER (PARTITION BY BI.Player_Name) AS 'Dominant Foot',
	MAX(CASE 
		WHEN DD.Aerial_Battles_Lost > DD.Aerial_Battles_Won THEN 'Short' 
		WHEN DD.Aerial_Battles_Lost < DD.Aerial_Battles_Won THEN 'Tall'
		WHEN DD.Aerial_Battles_Lost = DD.Aerial_Battles_Won THEN '6"'
		ELSE 'None' END)
		OVER (PARTITION BY BI.Player_Name) AS 'Height',
	AD.Goals, AD.Goals_Per_Match, 
	AD.Headed_Goals, AD.Goals_with_Right_Foot, AD.Goals_with_Left_Foot, 
	AD.Penalties_Scored, AD.Freekicks_Scored, AD.Shots, AD.Shots_on_Target, 
	AD.Shooting_Accuracy, AD.Hit_Woodwork, AD.Big_Chances_Missed, 
	DD.Clean_Sheets,DD.Goals_Conceded, DD.Tackles, DD.Tackle_Success, 
	DD.Last_Man_Tackles, DD.Blocked_Shots, DD.Interceptions, DD.Clearances,
	DD.Headed_Clearances, 
	MAX(CASE
		WHEN DD.Clearances > 1 THEN DD.Headed_Clearances / DD.Clearances 
		ELSE 0 END)
		OVER (PARTITION BY BI.Player_Name) AS 'Aerial Clearances',
	DD.Clearances_Off_Line, DD.Recoveries, DD.Duels_Won,
	DD.Duels_Lost, DD.Successful_50_50_s, DD.Aerial_Battles_Lost, 
	DD.Aerial_Battles_Won, DD.Own_Goals, DD.Errors_Leading_to_Goal
	
FROM BasicInfo BI
LEFT JOIN AttackData AD
ON AD.Player_Name = BI.Player_Name AND AD.Season = BI.Seasons
LEFT JOIN DefenceData DD
ON BI.Player_Name = DD.Player_Name AND BI.Seasons = DD.Season
LEFT JOIN PositionData PD
ON BI.Player_Name = PD.Player_Name
WHERE 
	COALESCE(Clean_Sheets, Goals_Conceded, Tackles, 
	Last_Man_Tackles, Blocked_Shots, Interceptions,
	Clearances, Headed_Clearances, Clearances_Off_Line,
	Recoveries, Duels_Won, Duels_Lost, Goals,
	Goals_Per_Match, Headed_Goals, Goals_with_Right_Foot, 
	Goals_with_Left_Foot, Penalties_Scored,
	Freekicks_Scored, Shots) IS NOT NULL



