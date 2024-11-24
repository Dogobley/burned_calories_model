#### Introduction

Loss-weight is a recurrent reason for train at gym. Excluding the diet, calories burned are important for that matter, but we can't know exactly how many calories are burned without expensive equipment. This model try to approach a good fit aproximation of the calories that are burned on the training based on:

## To Predict

- Calories_Burned: Total calories burned during each session.


## Features

- Age: Age of the gym member.
- Gender: Gender of the gym member (Male or Female).
- Weight (kg): Member’s weight in kilograms.
- Height (m): Member’s height in meters.
- Max_BPM: Maximum heart rate (beats per minute) during workout sessions.
- Avg_BPM: Average heart rate during workout sessions.
- Resting_BPM: Heart rate at rest before workout.
- Session_Duration (hours): Duration of each workout session in hours.
- Workout_Type: Type of workout performed (e.g., Cardio, Strength, Yoga, HIIT).
- Fat_Percentage: Body fat percentage of the member.
- Water_Intake (liters): Daily water intake during workouts.
- Workout_Frequency (days/week): Number of workout sessions per week.
- Experience_Level: Level of experience, from beginner (1) to expert (3).
- BMI: Body Mass Index, calculated from height and weight.

Data set on https://www.kaggle.com/api/v1/datasets/download/valakhorasani/gym-members-exercise-dataset
Kaggle page https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset?select=gym_members_exercise_tracking.csv

#### How to use

1. run: docker build -t burned_c_prediction .
2. run: docker run -it -p 9696:9696 burned_c_prediction:latest
3. Send a request as the following image
    - POST request
    - 0.0.0.0:9696 url
    - raw JSON

![alt text]("how to use.png")

Here a input example to copy:

{"age": 35,
 "gender": "Female",
 "weight_(kg)": 102.5,
 "height_(m)": 1.94,
 "max_bpm": 183,
 "avg_bpm": 158,
 "resting_bpm": 64,
 "session_duration_(hours)": 0.84,
 "workout_type": "Cardio",
 "fat_percentage": 21.1,
 "water_intake_(liters)": 2.4,
 "workout_frequency_(days/week)": 2,
 "experience_level": 1,
 "bmi": 27.23}