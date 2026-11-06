import pandas as pd
from pathlib import Path

data = []

def add(category, relation, pairs):
    for s1, s2 in pairs:
        data.append({
            "sentence1": s1,
            "sentence2": s2,
            "category": category,
            "expected_relation": relation,
            "human_similarity_score": "",
            "notes": ""
        })


add("paraphrase", "meaning_preserved", [
    ("The boy is riding a bicycle.", "A boy is cycling."),
    ("The woman purchased a new laptop.", "The woman bought a new laptop."),
    ("The meeting begins at nine o'clock.", "The meeting starts at 9 AM."),
    ("The dog is sleeping on the sofa.", "A dog is asleep on the couch."),
    ("She completed the assignment yesterday.", "She finished the assignment yesterday."),
    ("The restaurant is located near the station.", "The restaurant is close to the station."),
    ("He entered the room quietly.", "He quietly walked into the room."),
    ("The children are playing in the park.", "Kids are playing at the park."),
    ("The flight was cancelled because of bad weather.", "Bad weather caused the flight to be cancelled."),
    ("The company announced its results on Friday.", "On Friday, the company released its results.")
])


add("synonym_substitution", "meaning_preserved", [
    ("The movie was very good.", "The movie was very enjoyable."),
    ("The room was extremely small.", "The room was extremely tiny."),
    ("He gave a quick response.", "He gave a rapid response."),
    ("The task was difficult.", "The task was challenging."),
    ("She looked happy.", "She looked joyful."),
    ("The car is very fast.", "The car is very quick."),
    ("They live in a large house.", "They live in a big house."),
    ("The explanation was clear.", "The explanation was understandable."),
    ("She was afraid of the storm.", "She was scared of the storm."),
    ("The purchase was inexpensive.", "The purchase was cheap.")
])


add("negation", "meaning_changed", [
    ("The train arrived on time.", "The train did not arrive on time."),
    ("She likes coffee.", "She does not like coffee."),
    ("The door is open.", "The door is not open."),
    ("He attended the meeting.", "He did not attend the meeting."),
    ("The experiment was successful.", "The experiment was not successful."),
    ("They approved the proposal.", "They did not approve the proposal."),
    ("The shop is closed.", "The shop is not closed."),
    ("The patient recovered.", "The patient did not recover."),
    ("The package arrived today.", "The package did not arrive today."),
    ("The answer is correct.", "The answer is not correct.")
])


add("named_entity_substitution", "meaning_changed", [
    ("Alice travelled to Sydney.", "Alice travelled to Melbourne."),
    ("John works at Google.", "John works at Microsoft."),
    ("The conference was held in Paris.", "The conference was held in London."),
    ("Emma visited Japan last year.", "Emma visited Korea last year."),
    ("The match was played in Brisbane.", "The match was played in Perth."),
    ("David studies at Harvard University.", "David studies at Stanford University."),
    ("The company opened an office in Singapore.", "The company opened an office in Tokyo."),
    ("Michael met Sarah yesterday.", "Michael met Jessica yesterday."),
    ("The flight departed from Sydney Airport.", "The flight departed from Melbourne Airport."),
    ("The report was published by Reuters.", "The report was published by Bloomberg.")
])


add("number_date_substitution", "meaning_changed", [
    ("The meeting is on Monday.", "The meeting is on Tuesday."),
    ("The flight leaves at 8 AM.", "The flight leaves at 10 AM."),
    ("She bought three books.", "She bought five books."),
    ("The building has twenty floors.", "The building has thirty floors."),
    ("The event occurred in 2024.", "The event occurred in 2025."),
    ("The ticket costs fifty dollars.", "The ticket costs eighty dollars."),
    ("He scored 90 percent.", "He scored 60 percent."),
    ("The journey takes two hours.", "The journey takes four hours."),
    ("The class begins on 12 May.", "The class begins on 15 May."),
    ("The company hired 100 employees.", "The company hired 200 employees.")
])


add("antonym_substitution", "meaning_changed", [
    ("The water is hot.", "The water is cold."),
    ("The building is tall.", "The building is short."),
    ("The road is wide.", "The road is narrow."),
    ("The exam was easy.", "The exam was difficult."),
    ("The room was bright.", "The room was dark."),
    ("The car is expensive.", "The car is cheap."),
    ("The box is heavy.", "The box is light."),
    ("The process was fast.", "The process was slow."),
    ("The answer is correct.", "The answer is incorrect."),
    ("The result was positive.", "The result was negative.")
])


add("causal_relation_reversal", "meaning_changed", [
    ("Rain caused flooding.", "Flooding caused rain."),
    ("The storm caused the power outage.", "The power outage caused the storm."),
    ("Smoking increases the risk of disease.", "Disease increases the risk of smoking."),
    ("The fire caused the alarm to activate.", "The alarm caused the fire to activate."),
    ("Heavy traffic caused the delay.", "The delay caused heavy traffic."),
    ("The collision caused road congestion.", "Road congestion caused the collision."),
    ("The software bug caused the system failure.", "The system failure caused the software bug."),
    ("The earthquake caused the building to collapse.", "The building collapse caused the earthquake."),
    ("Low temperatures caused the water to freeze.", "Frozen water caused the low temperatures."),
    ("The economic crisis caused unemployment to rise.", "Rising unemployment caused the economic crisis.")
])


add("high_lexical_overlap_different_meaning", "meaning_changed", [
    ("The dog chased the cat.", "The cat chased the dog."),
    ("Sarah called Emma.", "Emma called Sarah."),
    ("The teacher praised the student.", "The student praised the teacher."),
    ("The doctor examined the patient.", "The patient examined the doctor."),
    ("The police arrested the thief.", "The thief arrested the police."),
    ("The manager interviewed the applicant.", "The applicant interviewed the manager."),
    ("The child followed the parent.", "The parent followed the child."),
    ("Tom defeated Jack.", "Jack defeated Tom."),
    ("The company acquired the startup.", "The startup acquired the company."),
    ("The truck hit the car.", "The car hit the truck.")
])


df = pd.DataFrame(data)
df.insert(0, "pair_id", range(1, len(df) + 1))

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "data" / "constructed"
out_dir.mkdir(parents=True, exist_ok=True)

output = out_dir / "constructed_pilot.csv"
df.to_csv(output, index=False)

print("Created:", output)
print("Total pairs:", len(df))

print("\nCategory counts:")
print(df["category"].value_counts().sort_index())

print("\nExpected relation counts:")
print(df["expected_relation"].value_counts())