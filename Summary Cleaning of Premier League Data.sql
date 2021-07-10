WITH cte AS (
	SELECT 
		Player_Name, 
		Season,
		ROW_NUMBER() OVER (
			PARTITION BY
				Player_Name,
				Season
			ORDER BY
				Player_Name,
				Season
			) row_num
			FROM AttackData 
)
DELETE FROM cte
WHERE row_num > 1

DELETE FROM AttackData
WHERE Goals = 0 AND 
	Goals_Per_Match = 0 AND 
	Headed_Goals = 0 AND 
	Goals_with_Right_Foot = 0 AND 
	Goals_with_Left_Foot = 0 AND 
	Penalties_Scored = 0 AND
	Freekicks_Scored = 0 AND
	Shots = 0


UPDATE AttackData
SET Season = REPLACE(Season, Season, STUFF(Season, 6, 0, SUBSTRING(Season, 1, 2))) 
WHERE Season != '1999/00'

UPDATE AttackData
SET Season = REPLACE(Season, Season, '1999/2000')
WHERE Season = '1999/00'

WITH cte AS (
	SELECT 
		Player_Name, 
		Season,
		ROW_NUMBER() OVER (
			PARTITION BY
				Player_Name,
				Season
			ORDER BY
				Player_Name,
				Season
			) row_num
			FROM DefenceData 
)
DELETE FROM cte
WHERE row_num > 1

DELETE FROM DefenceData
WHERE Clean_Sheets = 0 AND 
	Goals_Conceded = 0 AND 
	Tackles = 0 AND 
	Last_Man_Tackles = 0 AND 
	Blocked_Shots = 0 AND 
	Interceptions = 0 AND
	Clearances = 0 AND
	Headed_Clearances = 0 AND
	Clearances_Off_Line = 0 AND
	Recoveries = 0 AND
	Duels_Won = 0 AND
	Duels_Lost = 0


UPDATE DefenceData
SET Season = REPLACE(Season, Season, STUFF(Season, 6, 0, SUBSTRING(Season, 1, 2))) 
WHERE Season != '1999/00'

UPDATE DefenceData
SET Season = REPLACE(Season, Season, '1999/2000')
WHERE Season = '1999/00'

WITH cte AS (
	SELECT 
		Player_Name, 
		Position,
		ROW_NUMBER() OVER (
			PARTITION BY
				Player_Name,
				Position
			ORDER BY
				Player_Name
			) row_num
			FROM PositionData 
)
DELETE FROM cte
WHERE row_num > 1

WITH cte AS (
	SELECT 
		Player_Name, 
		Seasons,
		Team,
		ROW_NUMBER() OVER (
			PARTITION BY
				Player_Name,
				Seasons
			ORDER BY
				Player_Name,
				Seasons
			) row_num
			FROM BasicInfo 
)
DELETE FROM cte
WHERE row_num > 1

ALTER TABLE BasicInfo
ADD Number int identity(1,1)

