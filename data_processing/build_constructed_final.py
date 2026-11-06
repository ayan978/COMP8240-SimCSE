import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pilot_path = ROOT / "data" / "constructed" / "constructed_pilot_annotated.csv"
output_path = ROOT / "data" / "constructed" / "constructed_final_400.csv"

pilot = pd.read_csv(pilot_path)
pilot["source"] = "pilot_manual"

new_rows = []

def add(category, relation, pairs):
    for s1, s2 in pairs:
        new_rows.append({
            "sentence1": s1,
            "sentence2": s2,
            "category": category,
            "expected_relation": relation,
            "human_similarity_score": "",
            "notes": "",
            "source": "expanded_controlled",
        })

# ------------------------------------------------------------------
# 1. PARAPHRASE — 40 new
# ------------------------------------------------------------------
people_objects = [
    ("Maria", "a wireless keyboard"),
    ("Daniel", "a train ticket"),
    ("Sophie", "a winter coat"),
    ("James", "a coffee machine"),
    ("Olivia", "a concert ticket"),
]
for p, o in people_objects:
    add("paraphrase", "meaning_preserved",
        [(f"{p} purchased {o}.", f"{p} bought {o}.")])

events_times = [
    ("lecture", "10 AM"), ("workshop", "2 PM"), ("ceremony", "6 PM"),
    ("interview", "11 AM"), ("training session", "3 PM"),
]
for e, t in events_times:
    add("paraphrase", "meaning_preserved",
        [(f"The {e} begins at {t}.", f"The {e} starts at {t}.")])

locations = [
    ("hotel", "airport"), ("library", "university"),
    ("cafe", "museum"), ("clinic", "shopping centre"),
    ("stadium", "train station"),
]
for p, l in locations:
    add("paraphrase", "meaning_preserved",
        [(f"The {p} is located near the {l}.",
          f"The {p} is close to the {l}.")])

tasks = [
    ("Anna", "report", "this morning"),
    ("Peter", "presentation", "last night"),
    ("Rachel", "project", "on Monday"),
    ("Thomas", "application", "yesterday"),
    ("Emily", "survey", "this afternoon"),
]
for p, task, when in tasks:
    add("paraphrase", "meaning_preserved",
        [(f"{p} completed the {task} {when}.",
          f"{p} finished the {task} {when}.")])

entries = [
    ("Mark", "office"), ("Laura", "classroom"), ("Henry", "hall"),
    ("Nina", "laboratory"), ("George", "building"),
]
for p, place in entries:
    add("paraphrase", "meaning_preserved",
        [(f"{p} entered the {place} quietly.",
          f"{p} quietly walked into the {place}.")])

cancelled = [
    ("outdoor concert", "heavy rain"),
    ("football match", "a severe storm"),
    ("ferry service", "strong winds"),
    ("school excursion", "bad weather"),
    ("festival", "flooding"),
]
for event, cause in cancelled:
    add("paraphrase", "meaning_preserved",
        [(f"The {event} was cancelled because of {cause}.",
          f"{cause.capitalize()} caused the {event} to be cancelled.")])

groups_places = [
    ("students", "courtyard"), ("children", "playground"),
    ("friends", "garden"), ("players", "field"),
    ("tourists", "beach"),
]
for group, place in groups_places:
    add("paraphrase", "meaning_preserved",
        [(f"The {group} are playing in the {place}.",
          f"The {group} are playing at the {place}.")])

announcements = [
    ("bank", "quarterly results", "Monday"),
    ("university", "new policy", "Tuesday"),
    ("airline", "new route", "Wednesday"),
    ("company", "annual results", "Thursday"),
    ("government agency", "new guidelines", "Friday"),
]
for org, item, day in announcements:
    add("paraphrase", "meaning_preserved",
        [(f"The {org} announced its {item} on {day}.",
          f"On {day}, the {org} released its {item}.")])

# ------------------------------------------------------------------
# 2. SYNONYM SUBSTITUTION — 40 new
# ------------------------------------------------------------------
synonym_groups = [
    ("rapid", "quick",
     ["response", "recovery", "decision", "change", "improvement"]),
    ("large", "big",
     ["garden", "office", "warehouse", "kitchen", "apartment"]),
    ("difficult", "challenging",
     ["exam", "project", "problem", "course", "assignment"]),
    ("happy", "joyful",
     ["child", "teacher", "winner", "family", "couple"]),
    ("afraid", "scared",
     ["student", "driver", "child", "tourist", "passenger"]),
    ("inexpensive", "cheap",
     ["meal", "ticket", "hotel", "phone", "service"]),
    ("clear", "understandable",
     ["instruction", "message", "guidance", "answer", "description"]),
    ("silent", "quiet",
     ["room", "library", "street", "office", "hallway"]),
]

for a, b, nouns in synonym_groups:
    for noun in nouns:
        if a in {"happy", "joyful", "afraid", "scared"}:
            s1 = f"The {noun} looked {a}."
            s2 = f"The {noun} looked {b}."
        else:
            s1 = f"The {noun} was {a}."
            s2 = f"The {noun} was {b}."
        add("synonym_substitution", "meaning_preserved", [(s1, s2)])

# ------------------------------------------------------------------
# 3. NEGATION — 40 new
# ------------------------------------------------------------------
negation_sets = [
    ("The document is available online.",
     "The document is not available online."),
    ("The room is available today.",
     "The room is not available today."),
    ("The product is available locally.",
     "The product is not available locally."),
    ("The service is available overnight.",
     "The service is not available overnight."),
    ("The course is available this semester.",
     "The course is not available this semester."),

    ("Sarah attended the workshop.",
     "Sarah did not attend the workshop."),
    ("Michael attended the lecture.",
     "Michael did not attend the lecture."),
    ("Emma attended the conference.",
     "Emma did not attend the conference."),
    ("David attended the ceremony.",
     "David did not attend the ceremony."),
    ("Lisa attended the interview.",
     "Lisa did not attend the interview."),

    ("The committee approved the budget.",
     "The committee did not approve the budget."),
    ("The board approved the merger.",
     "The board did not approve the merger."),
    ("The council approved the plan.",
     "The council did not approve the plan."),
    ("The manager approved the request.",
     "The manager did not approve the request."),
    ("The university approved the application.",
     "The university did not approve the application."),

    ("The bus arrived early.",
     "The bus did not arrive early."),
    ("The flight arrived yesterday.",
     "The flight did not arrive yesterday."),
    ("The parcel arrived safely.",
     "The parcel did not arrive safely."),
    ("The ambulance arrived quickly.",
     "The ambulance did not arrive quickly."),
    ("The shipment arrived this morning.",
     "The shipment did not arrive this morning."),

    ("The box contains the documents.",
     "The box does not contain the documents."),
    ("The report contains the results.",
     "The report does not contain the results."),
    ("The package contains a charger.",
     "The package does not contain a charger."),
    ("The file contains personal information.",
     "The file does not contain personal information."),
    ("The folder contains the images.",
     "The folder does not contain the images."),

    ("James passed the exam.",
     "James did not pass the exam."),
    ("Maria passed the driving test.",
     "Maria did not pass the driving test."),
    ("Henry passed the assessment.",
     "Henry did not pass the assessment."),
    ("Alice passed the final test.",
     "Alice did not pass the final test."),
    ("George passed the course.",
     "George did not pass the course."),

    ("The museum opened on Monday.",
     "The museum did not open on Monday."),
    ("The store opened at nine.",
     "The store did not open at nine."),
    ("The office opened yesterday.",
     "The office did not open yesterday."),
    ("The restaurant opened this week.",
     "The restaurant did not open this week."),
    ("The gate opened automatically.",
     "The gate did not open automatically."),

    ("The software supports offline access.",
     "The software does not support offline access."),
    ("The phone supports wireless charging.",
     "The phone does not support wireless charging."),
    ("The system supports multiple users.",
     "The system does not support multiple users."),
    ("The browser supports this feature.",
     "The browser does not support this feature."),
    ("The application supports dark mode.",
     "The application does not support dark mode."),
]
add("negation", "meaning_changed", negation_sets)

# ------------------------------------------------------------------
# 4. NAMED ENTITY SUBSTITUTION — 40 new
# ------------------------------------------------------------------
entity_sets = [
    ("Liam travelled to Adelaide.", "Liam travelled to Darwin."),
    ("Sophia travelled to Rome.", "Sophia travelled to Madrid."),
    ("Noah travelled to Toronto.", "Noah travelled to Vancouver."),
    ("Grace travelled to Berlin.", "Grace travelled to Vienna."),
    ("Ethan travelled to Auckland.", "Ethan travelled to Wellington."),

    ("Rachel works at Amazon.", "Rachel works at Apple."),
    ("Daniel works at IBM.", "Daniel works at Oracle."),
    ("Jessica works at Tesla.", "Jessica works at Ford."),
    ("Oliver works at Meta.", "Oliver works at Netflix."),
    ("Hannah works at Samsung.", "Hannah works at Sony."),

    ("Lucas studies at Oxford University.", "Lucas studies at Cambridge University."),
    ("Amelia studies at Yale University.", "Amelia studies at Princeton University."),
    ("Jack studies at Monash University.", "Jack studies at the University of Melbourne."),
    ("Mia studies at UNSW.", "Mia studies at the University of Sydney."),
    ("William studies at MIT.", "William studies at Caltech."),

    ("The article was published by CNN.", "The article was published by BBC."),
    ("The story was published by The Guardian.", "The story was published by The Times."),
    ("The report was published by ABC News.", "The report was published by SBS News."),
    ("The interview was published by Forbes.", "The interview was published by Bloomberg."),
    ("The investigation was published by Reuters.", "The investigation was published by AP."),

    ("The flight departed from Brisbane.", "The flight departed from Adelaide."),
    ("The flight departed from Dubai.", "The flight departed from Doha."),
    ("The flight departed from Singapore.", "The flight departed from Bangkok."),
    ("The flight departed from Los Angeles.", "The flight departed from San Francisco."),
    ("The flight departed from London.", "The flight departed from Manchester."),

    ("The summit was held in Geneva.", "The summit was held in Brussels."),
    ("The exhibition was held in New York.", "The exhibition was held in Chicago."),
    ("The ceremony was held in Canberra.", "The ceremony was held in Sydney."),
    ("The conference was held in Seoul.", "The conference was held in Beijing."),
    ("The workshop was held in Munich.", "The workshop was held in Frankfurt."),

    ("Emma visited Thailand last year.", "Emma visited Vietnam last year."),
    ("Jacob visited France last year.", "Jacob visited Italy last year."),
    ("Charlotte visited Canada last year.", "Charlotte visited Mexico last year."),
    ("Benjamin visited India last year.", "Benjamin visited Nepal last year."),
    ("Ella visited Norway last year.", "Ella visited Sweden last year."),

    ("Andrew met Olivia yesterday.", "Andrew met Chloe yesterday."),
    ("Sophie met Daniel yesterday.", "Sophie met James yesterday."),
    ("Henry met Alice yesterday.", "Henry met Maria yesterday."),
    ("Jack met Emma yesterday.", "Jack met Sophia yesterday."),
    ("Thomas met Rachel yesterday.", "Thomas met Hannah yesterday."),
]
add("named_entity_substitution", "meaning_changed", entity_sets)

# ------------------------------------------------------------------
# 5. NUMBER / DATE SUBSTITUTION — 40 new
# ------------------------------------------------------------------
number_sets = [
    ("The appointment is on Wednesday.", "The appointment is on Thursday."),
    ("The exam is on Friday.", "The exam is on Saturday."),
    ("The ceremony is on Sunday.", "The ceremony is on Monday."),
    ("The interview is on Tuesday.", "The interview is on Thursday."),
    ("The workshop is on Thursday.", "The workshop is on Friday."),

    ("The train leaves at 6 AM.", "The train leaves at 9 AM."),
    ("The bus leaves at 7 AM.", "The bus leaves at 11 AM."),
    ("The ferry leaves at 1 PM.", "The ferry leaves at 4 PM."),
    ("The class starts at 8 AM.", "The class starts at 10 AM."),
    ("The concert starts at 6 PM.", "The concert starts at 8 PM."),

    ("She bought four notebooks.", "She bought seven notebooks."),
    ("He ordered two pizzas.", "He ordered six pizzas."),
    ("They purchased five chairs.", "They purchased nine chairs."),
    ("The teacher collected eight assignments.", "The teacher collected twelve assignments."),
    ("The shop sold ten bicycles.", "The shop sold fifteen bicycles."),

    ("The tower has fifteen floors.", "The tower has twenty-five floors."),
    ("The hotel has twelve floors.", "The hotel has eighteen floors."),
    ("The office has eight floors.", "The office has sixteen floors."),
    ("The hospital has ten floors.", "The hospital has fourteen floors."),
    ("The apartment block has six floors.", "The apartment block has eleven floors."),

    ("The agreement was signed in 2021.", "The agreement was signed in 2023."),
    ("The bridge opened in 2018.", "The bridge opened in 2020."),
    ("The company was founded in 2005.", "The company was founded in 2010."),
    ("The study was published in 2019.", "The study was published in 2022."),
    ("The building was completed in 2016.", "The building was completed in 2021."),

    ("The meal costs thirty dollars.", "The meal costs sixty dollars."),
    ("The pass costs forty dollars.", "The pass costs seventy dollars."),
    ("The jacket costs eighty dollars.", "The jacket costs one hundred dollars."),
    ("The course costs five hundred dollars.", "The course costs eight hundred dollars."),
    ("The phone costs six hundred dollars.", "The phone costs nine hundred dollars."),

    ("She scored 85 percent.", "She scored 65 percent."),
    ("He scored 72 percent.", "He scored 92 percent."),
    ("The team achieved 80 percent.", "The team achieved 50 percent."),
    ("The student scored 95 percent.", "The student scored 75 percent."),
    ("The model achieved 88 percent.", "The model achieved 68 percent."),

    ("The trip takes three hours.", "The trip takes six hours."),
    ("The flight takes seven hours.", "The flight takes ten hours."),
    ("The hike takes four hours.", "The hike takes eight hours."),
    ("The procedure takes one hour.", "The procedure takes three hours."),
    ("The journey takes five hours.", "The journey takes nine hours."),
]
add("number_date_substitution", "meaning_changed", number_sets)

# ------------------------------------------------------------------
# 6. ANTONYM SUBSTITUTION — 40 new
# ------------------------------------------------------------------
antonym_sets = [
    ("The soup is warm.", "The soup is cold."),
    ("The coffee is hot.", "The coffee is cold."),
    ("The surface is smooth.", "The surface is rough."),
    ("The corridor is wide.", "The corridor is narrow."),
    ("The ceiling is high.", "The ceiling is low."),
    ("The bag is heavy.", "The bag is light."),
    ("The fabric is soft.", "The fabric is hard."),
    ("The street is busy.", "The street is quiet."),
    ("The room is clean.", "The room is dirty."),
    ("The chair is comfortable.", "The chair is uncomfortable."),

    ("The lecture was interesting.", "The lecture was boring."),
    ("The instructions were simple.", "The instructions were complicated."),
    ("The solution was correct.", "The solution was incorrect."),
    ("The outcome was positive.", "The outcome was negative."),
    ("The service was excellent.", "The service was terrible."),
    ("The journey was short.", "The journey was long."),
    ("The response was fast.", "The response was slow."),
    ("The task was easy.", "The task was difficult."),
    ("The building was old.", "The building was new."),
    ("The container was full.", "The container was empty."),

    ("The market is rising.", "The market is falling."),
    ("The temperature is increasing.", "The temperature is decreasing."),
    ("The door is open.", "The door is closed."),
    ("The machine is active.", "The machine is inactive."),
    ("The account is enabled.", "The account is disabled."),
    ("The path is safe.", "The path is dangerous."),
    ("The device is connected.", "The device is disconnected."),
    ("The answer is true.", "The answer is false."),
    ("The patient is awake.", "The patient is asleep."),
    ("The signal is strong.", "The signal is weak."),

    ("The company reported a profit.", "The company reported a loss."),
    ("Demand increased sharply.", "Demand decreased sharply."),
    ("The population grew rapidly.", "The population declined rapidly."),
    ("The team won the match.", "The team lost the match."),
    ("The price rose yesterday.", "The price fell yesterday."),
    ("The light became brighter.", "The light became darker."),
    ("The road became wider.", "The road became narrower."),
    ("The queue became longer.", "The queue became shorter."),
    ("The situation improved.", "The situation worsened."),
    ("The noise became louder.", "The noise became quieter."),
]
add("antonym_substitution", "meaning_changed", antonym_sets)

# ------------------------------------------------------------------
# 7. CAUSAL RELATION REVERSAL — 40 new
# ------------------------------------------------------------------
causal_pairs = [
    ("Heavy rainfall caused the river to overflow.", "The overflowing river caused the heavy rainfall."),
    ("A power surge caused the computer to shut down.", "The computer shutdown caused the power surge."),
    ("The drought caused crop production to decline.", "Declining crop production caused the drought."),
    ("The accident caused traffic to stop.", "Stopped traffic caused the accident."),
    ("High demand caused prices to increase.", "Higher prices caused the high demand."),
    ("The infection caused the patient's fever.", "The patient's fever caused the infection."),
    ("The broken pipe caused the floor to flood.", "The flooded floor caused the pipe to break."),
    ("Strong winds caused the tree to fall.", "The fallen tree caused the strong winds."),
    ("The battery failure caused the device to turn off.", "The device turning off caused the battery failure."),
    ("The road closure caused a long delay.", "The long delay caused the road closure."),

    ("The software update caused the application to restart.", "The application restart caused the software update."),
    ("Heavy snow caused the airport to close.", "The airport closure caused the heavy snow."),
    ("The engine failure caused the vehicle to stop.", "The stopped vehicle caused the engine failure."),
    ("A lack of sleep caused her fatigue.", "Her fatigue caused the lack of sleep."),
    ("The damaged cable caused the network outage.", "The network outage caused the cable damage."),
    ("Poor visibility caused the flight delay.", "The flight delay caused the poor visibility."),
    ("The heat caused the ice to melt.", "The melting ice caused the heat."),
    ("The alarm caused the building to be evacuated.", "The evacuation caused the alarm."),
    ("A technical fault caused the train cancellation.", "The train cancellation caused the technical fault."),
    ("Heavy rain caused the match to be postponed.", "The postponed match caused the heavy rain."),

    ("The medication caused the patient's drowsiness.", "The patient's drowsiness caused the medication."),
    ("The shortage caused food prices to rise.", "Rising food prices caused the shortage."),
    ("The storm caused several flights to be cancelled.", "The cancelled flights caused the storm."),
    ("The fire caused smoke to fill the room.", "The smoke-filled room caused the fire."),
    ("Low demand caused production to decrease.", "Lower production caused the low demand."),
    ("The collision caused the vehicle damage.", "The vehicle damage caused the collision."),
    ("The new policy caused operating costs to fall.", "Lower operating costs caused the new policy."),
    ("The rainfall caused the soil to become wet.", "The wet soil caused the rainfall."),
    ("The earthquake caused several roads to crack.", "The cracked roads caused the earthquake."),
    ("The outage caused the website to become unavailable.", "The unavailable website caused the outage."),

    ("The cold weather caused the lake to freeze.", "The frozen lake caused the cold weather."),
    ("The faulty sensor caused the warning light to activate.", "The warning light caused the sensor fault."),
    ("The strike caused the factory to close.", "The factory closure caused the strike."),
    ("The leak caused water pressure to drop.", "Low water pressure caused the leak."),
    ("The recession caused unemployment to increase.", "Higher unemployment caused the recession."),
    ("The explosion caused nearby windows to break.", "The broken windows caused the explosion."),
    ("The illness caused him to miss work.", "Missing work caused his illness."),
    ("The construction work caused traffic congestion.", "Traffic congestion caused the construction work."),
    ("The server failure caused the service interruption.", "The service interruption caused the server failure."),
    ("The temperature drop caused frost to form.", "The frost caused the temperature to drop."),
]
add("causal_relation_reversal", "meaning_changed", causal_pairs)

# ------------------------------------------------------------------
# 8. HIGH LEXICAL OVERLAP / DIFFERENT MEANING — 40 new
# ------------------------------------------------------------------
role_reversals = [
    ("The lawyer questioned the witness.", "The witness questioned the lawyer."),
    ("The journalist interviewed the actor.", "The actor interviewed the journalist."),
    ("The nurse helped the patient.", "The patient helped the nurse."),
    ("The guard stopped the visitor.", "The visitor stopped the guard."),
    ("The chef thanked the waiter.", "The waiter thanked the chef."),
    ("The coach congratulated the player.", "The player congratulated the coach."),
    ("The officer warned the driver.", "The driver warned the officer."),
    ("The student corrected the teacher.", "The teacher corrected the student."),
    ("The customer called the manager.", "The manager called the customer."),
    ("The editor contacted the writer.", "The writer contacted the editor."),

    ("The lion chased the zebra.", "The zebra chased the lion."),
    ("The cat followed the bird.", "The bird followed the cat."),
    ("The horse kicked the farmer.", "The farmer kicked the horse."),
    ("The dog frightened the child.", "The child frightened the dog."),
    ("The bee stung the gardener.", "The gardener stung the bee."),

    ("Alice invited Robert.", "Robert invited Alice."),
    ("Emily thanked Daniel.", "Daniel thanked Emily."),
    ("James called Sophia.", "Sophia called James."),
    ("Michael greeted Hannah.", "Hannah greeted Michael."),
    ("Olivia followed Ethan.", "Ethan followed Olivia."),

    ("The bus hit the bicycle.", "The bicycle hit the bus."),
    ("The motorcycle passed the truck.", "The truck passed the motorcycle."),
    ("The train struck the car.", "The car struck the train."),
    ("The taxi followed the van.", "The van followed the taxi."),
    ("The boat approached the ship.", "The ship approached the boat."),

    ("The university funded the laboratory.", "The laboratory funded the university."),
    ("The bank acquired the company.", "The company acquired the bank."),
    ("The supplier contacted the retailer.", "The retailer contacted the supplier."),
    ("The agency hired the consultant.", "The consultant hired the agency."),
    ("The publisher sued the author.", "The author sued the publisher."),

    ("The prosecutor accused the defendant.", "The defendant accused the prosecutor."),
    ("The judge questioned the lawyer.", "The lawyer questioned the judge."),
    ("The employer promoted the employee.", "The employee promoted the employer."),
    ("The parent praised the child.", "The child praised the parent."),
    ("The doctor advised the patient.", "The patient advised the doctor."),

    ("The researcher cited the professor.", "The professor cited the researcher."),
    ("The director criticised the actor.", "The actor criticised the director."),
    ("The buyer paid the seller.", "The seller paid the buyer."),
    ("The landlord contacted the tenant.", "The tenant contacted the landlord."),
    ("The company hired the engineer.", "The engineer hired the company."),
]
add("high_lexical_overlap_different_meaning", "meaning_changed", role_reversals)

# ------------------------------------------------------------------
# Combine + validate
# ------------------------------------------------------------------
new_df = pd.DataFrame(new_rows)

assert len(pilot) == 80, f"Expected 80 pilot rows, found {len(pilot)}"
assert len(new_df) == 320, f"Expected 320 new rows, found {len(new_df)}"

final = pd.concat([pilot, new_df], ignore_index=True)
final["pair_id"] = range(1, len(final) + 1)

ordered_cols = [
    "pair_id", "sentence1", "sentence2", "category",
    "expected_relation", "human_similarity_score", "notes", "source"
]
final = final[ordered_cols]

assert len(final) == 400

counts = final["category"].value_counts()
assert (counts == 50).all(), counts

# Detect exact or reversed duplicate pairs across rows.
normalised = final.apply(
    lambda r: " || ".join(sorted([
        str(r["sentence1"]).strip().lower(),
        str(r["sentence2"]).strip().lower()
    ])),
    axis=1
)
assert normalised.duplicated().sum() == 0, "Duplicate/reversed duplicate pair detected"

final.to_csv(output_path, index=False)

print("Created:", output_path)
print("Total pairs:", len(final))
print("\nCategory counts:")
print(final["category"].value_counts().sort_index())
print("\nExpected relation counts:")
print(final["expected_relation"].value_counts())
print("\nSource counts:")
print(final["source"].value_counts())
print("\nMissing sentence values:",
      final[["sentence1", "sentence2"]].isna().sum().sum())
print("Duplicate/reversed duplicate pairs:", normalised.duplicated().sum())
