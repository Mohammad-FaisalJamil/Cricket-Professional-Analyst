import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Cricket Match Predictor", version="1.0.0")

class MatchRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    month: int = 9

class PredictionResponse(BaseModel):
    innings: int
    batting: str
    bowling: str
    predicted_runs: int
    predicted_wickets: int

def predict_cricket_match(team1, team2, venue, month=9):
    """
    Predict cricket match results for any two teams at any venue
    
    Args:
        team1: First team name
        team2: Second team name  
        venue: Venue name
        month: Month of match (default 9 for September)
    """
    
    # Load models and dataset
    with open('cricket_runs_model_final_tuned.pkl', 'rb') as f:
        runs_model = pickle.load(f)
    with open('cricket_wickets_model_final_tuned.pkl', 'rb') as f:
        wickets_model = pickle.load(f)
    
    df = pd.read_csv('cricket_features_enhanced_with_venue.csv')
    feature_cols = [col for col in df.columns if col not in ['total_runs', 'total_wickets']]
    
    # Get team data
    team1_data = df[df['batting_team'] == team1].iloc[-1] if len(df[df['batting_team'] == team1]) > 0 else None
    team2_data = df[df['batting_team'] == team2].iloc[-1] if len(df[df['batting_team'] == team2]) > 0 else None
    
    if team1_data is None or team2_data is None:
        return None
    
    # Get venue data
    venue_data = df[df['venue'] == venue].iloc[0] if len(df[df['venue'] == venue]) > 0 else None
    if venue_data is None:
        return None
    
    # Get country from venue data
    country = venue_data['country']
    
    predictions = []
    for innings in [1, 2]:
        for batting_team, bowling_team in [(team1, team2), (team2, team1)]:
            team_data = team1_data if batting_team == team1 else team2_data
            
            # Create prediction row
            pred_row = team_data.copy()
            
            # Update match-specific values
            pred_row['venue'] = venue
            pred_row['country'] = country
            pred_row['innings'] = innings
            pred_row['batting_team'] = batting_team
            pred_row['bowling_team'] = bowling_team
            pred_row['is_home'] = 'AWAY'  # Assuming neutral venue
            pred_row['start_month'] = month
            pred_row['batting_first'] = 1 if innings == 1 else 0
            pred_row['chasing_flag'] = 1 if innings == 2 else 0
            
            # Update venue-specific features if team has played at venue
            team_at_venue = df[(df['batting_team'] == batting_team) & (df['venue'] == venue)]
            if len(team_at_venue) > 0:
                pred_row['batting_team_venue_avg_runs_first'] = team_at_venue[team_at_venue['innings'] == 1]['total_runs'].mean() if len(team_at_venue[team_at_venue['innings'] == 1]) > 0 else pred_row['avg_runs_batting_first']
                pred_row['batting_team_venue_avg_runs_second'] = team_at_venue[team_at_venue['innings'] == 2]['total_runs'].mean() if len(team_at_venue[team_at_venue['innings'] == 2]) > 0 else pred_row['avg_runs_batting_second']
                pred_row['batting_team_venue_avg_wickets_lost'] = team_at_venue['total_wickets'].mean()
                pred_row['batting_team_venue_matches'] = len(team_at_venue)
            
            # Create DataFrame and predict
            pred_df = pd.DataFrame([pred_row])[feature_cols]
            runs_pred = runs_model.predict(pred_df)[0]
            wickets_pred = wickets_model.predict(pred_df)[0]
            
            predictions.append({
                'Innings': innings,
                'Batting': batting_team,
                'Bowling': bowling_team,
                'Predicted_Runs': round(runs_pred),
                'Predicted_Wickets': round(wickets_pred)
            })
    
    return predictions

@app.get("/")
def root():
    return {"message": "Cricket Match Predictor API"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_match(request: MatchRequest):
    try:
        predictions = predict_cricket_match(request.team1, request.team2, request.venue, request.month)
        if not predictions:
            raise HTTPException(status_code=404, detail="Team or venue data not found")
        
        return [
            PredictionResponse(
                innings=p['Innings'],
                batting=p['Batting'],
                bowling=p['Bowling'],
                predicted_runs=p['Predicted_Runs'],
                predicted_wickets=p['Predicted_Wickets']
            ) for p in predictions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)