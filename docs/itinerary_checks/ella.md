# Itinerary check: Ella

**Request:** Plan a 3-day trip to Ella.
**Gemini calls:** 3
**Tools that failed:** find_attractions
**Wikivoyage fallback for attractions:** used

## Steps
```
[human] Plan a 3-day trip to Ella.
[llm -> tool call] get_destination_guide({'destination': 'Ella', 'topic': 'all'})
[llm -> tool call] find_attractions({'destination': 'Ella', 'category': 'all'})
[llm -> tool call] get_weather_forecast({'destination': 'Ella'})
[llm -> tool call] estimate_travel_time({'destination': 'Ella', 'origin': 'Colombo'})
[tool result] Wikivoyage guide for Ella (all topics): ...
[tool result] Attractions unavailable for 'Ella': No Overpass server could complete the search. overpass-api.de: The Overpass API returned HTTP 502. overpass.private.coffee: The Overpass API did not respond within 30 seconds. ...
[tool result] Weather forecast for Ella Town (Badulla District, Uva Province, Sri Lanka), at lat 6.88, lon 81.05, elevation 1031 m. ...
[tool result] Road trip from Colombo (Colombo District, Western Province, Sri Lanka) to Ella Town (Badulla District, Uva Province, Sri Lanka): ...
[llm -> tool call] convert_currency({'to_currency': 'USD', 'amount': 3000, 'from_currency': 'LKR'})
[llm -> tool call] estimate_travel_time({'destination': 'Badulla', 'origin': 'Ella'})
[tool result] 3,000.00 LKR = 9.01 USD ...
[tool result] Road trip from Ella Town (Badulla District, Uva Province, Sri Lanka) to Badulla (Uva Province, Sri Lanka): ...
[llm -> final answer]
```

## Automatic checks
- [x] called get_destination_guide
- [x] called find_attractions
- [x] called get_weather_forecast
- [x] called estimate_travel_time
- [x] get_destination_guide asked about Ella (got ['Ella'])
- [x] find_attractions asked about Ella (got ['Ella'])
- [x] get_weather_forecast asked about Ella (got ['Ella'])
- [x] travel time from Colombo to Ella (got [('Colombo', 'Ella'), ('Ella', 'Badulla')])
- [x] plan says it assumed Colombo as the start
- [x] plan has a 'Not available' list
- [x] the run ended with a written plan

**Result:** 11/11 automatic checks passed

## Plan

Here is a suggested 3-day itinerary for Ella, assuming you are starting your trip from **Colombo** (as a starting point was not specified). 

---

### Getting to Ella
* **From Colombo:** ~203 km road distance, taking about **3 hours 42 minutes by car** (without traffic/stops, source: *OSRM*). *Note: Hill road drives, buses, or trains typically take longer.*
* **Local Getting Around:** Renting a scooter through your guest house or taking local tuk-tuks is recommended (source: *Wikivoyage*).

---

### Day-by-Day Itinerary

#### **Day 1: Little Adam's Peak & Green Tea Tasting**
* **Morning:** Walk east from Ella town center (approx. 20 minutes) along the main road to reach **Little Adam's Peak**. Enjoy an easy hike through tea plantations offering views of Ella Rock, Rawana Falls, and the Newburgh Estate (source: *Wikivoyage*). 
* **Afternoon:** Visit the nearby **Newburgh Green Tea Factory** (approx. 30 minutes' walk from town along Passara Road) for a quick factory tour and green tea tasting (source: *Wikivoyage*).
* **Evening:** Relax with a massage at a local massage place in town (source: *Wikivoyage*).
* **Weather (Day 1):** Min 16.0 °C, Max 25.9 °C; moderate drizzle (70% chance of rain, 2.4 mm rainfall, max wind 8.7 km/h) (source: *Open-Meteo*).

#### **Day 2: Nine Arch Bridge & Organic Tea Plantation Tour**
* **Morning:** Head early to the **Nine Arch Bridge** (about a 45-minute walk from the town center via Passara Road or walking along the train tracks from Ella station—be alert for trains). Watch trains cross the famous rock bridge (source: *Wikivoyage*).
* **Afternoon:** Visit the **Ella Organic Tea Garden**, located 5 km south of the town center off Kithalella Road. Take a guided plantation tour, learn about spice and fruit cultivation, and sample different teas.
  * **Price:** 3,000 LKR per person (approx. **$9.01 USD**, at an exchange rate of 1 LKR = 0.003003 USD; source: *Exchange Rate API*).
* **Weather (Day 2):** Min 16.0 °C, Max 26.5 °C; moderate rain showers (92% chance of rain, 8.3 mm rainfall, max wind 4.7 km/h) (source: *Open-Meteo*).

#### **Day 3: Day Trip to Badulla**
* **Morning/Afternoon:** Take a short trip to **Badulla**, located about **21 km** from Ella (**~28 minutes by car** without traffic, source: *OSRM*). Explore the surrounding Uva province countryside.
* **Safety Tip:** Avoid walking through rural vegetation in shorts and sandals, especially after rain, due to leeches. Carry hand sanitizer or Dettol to remove them safely (source: *Wikivoyage*).
* **Weather (Day 3):** Min 16.1 °C, Max 25.5 °C; moderate drizzle (56% chance of rain, 2.0 mm rainfall, max wind 7.0 km/h) (source: *Open-Meteo*).

---

### **Not Available**
* **Live OpenStreetMap Attractions:** OpenStreetMap servers were unavailable (502 error), so all listed sights come from the *Wikivoyage* guide fallback without exact straight-line distances.
* **Accommodation & Restaurant Prices:** Hotel stay costs, food prices, and specific hiking route maps are not provided by the tools.
* **Train Timetables:** Live train departure schedules between Ella and Badulla/Colombo are not available in the tool data.

## Your review (compare the plan with the tool results)
- [ ] Every sight in the plan appears in the find_attractions or guide results.
- [ ] Travel times quoted match the estimate_travel_time results (car, no traffic).
- [ ] Weather is given only for forecast days; other days say no forecast.
- [ ] Every price comes from the guide, with LKR/USD from convert_currency.
- [ ] Each fact names its source (OpenStreetMap, Wikivoyage, OSRM, Open-Meteo, rates).
- [ ] Failed tools and missing data are listed under 'Not available', not invented.
- [ ] The days are realistic: not too many sights a day, travel time accounted for.

## Tool results the plan must be based on

### get_destination_guide
```
Wikivoyage guide for Ella (all topics):
[Introduction] Ella (also spelt 'Elle') is a beautiful small town in Uva province on the southern edge of Sri Lanka's Hill Country. It is popular with tourists and makes a great base from which to explore the surrounding countryside. It's in the middle of beautiful countryside, with small vegetable plots in the valleys, tea plantations on the hill slopes and forests on the tops. ...
[Get in] By train: Cyclone Ditwah In November/December 2025, Cyclone Ditwah damaged 70% of the Sri Lankan railway network. Repair work is ongoing (as of February 2026), but several lines are still out of service: Trains from Badulla and Ella travel as far as Nanu Oya (close to Nuwara Eliya) but don't continue to Kandy or Colombo. ...
[Get around] There are plenty of tuk-tuks around, but the best way to get around is to rent a scooter- most guest houses do this. Online carhire services exist such as Ella Taxi Service.
[See] 1 Little Adam's Peak. A relatively easy hike to the east of the town center. Walk along the main road for 20 minutes to reach the path, cutting across tea plantations before the final climb up. Great views on Ella Rock and Rawana Falls. You can see the Newburgh Tea Estate on the other side (north). Busy during high season. There is a zip line near the bottom of the trail, available for a fee, though it won't save you time. 2 Nine Arch Bridge. ...
[Do] Have a Massage at one of the local massage places in town. 1 Ella Organic Tea Garden, 5 km south of Ella centre just off Kithalella road, ☏ +94 711344503. 09:00, 11:00, 13:00. Visit one of the few organic tea growers in Sri Lanka. You will get a short tour of the tea plantation, being introduced to several spices and fruits that grow there. After that, learn how the different types of tea are made and taste them all. Rs 3,000 per person. ...
[Stay healthy] Don't walk in the countryside in shorts and sandals or flipflops, especially after it's been raining. There are lots of leeches that hang on to the vegetation by the side of tracks waiting for a nice juicy mammal to walk by. Take some Dettol or hand sanitizer with you and spray it on the offending leech, which will drop off.
Sections ending in ... were shortened; full article: https://en.wikivoyage.org/wiki/Ella
Source: Wikivoyage, community-written (CC BY-SA 4.0); prices, timetables and transport details may be out of date.
```

### find_attractions
```
Attractions unavailable for 'Ella': No Overpass server could complete the search. overpass-api.de: The Overpass API returned HTTP 502. overpass.private.coffee: The Overpass API did not respond within 30 seconds.
FALLBACK: sights named in the Wikivoyage guide instead (no distances, not filtered by category).
Wikivoyage guide for Ella (see and do):
[See] 1 Little Adam's Peak. A relatively easy hike to the east of the town center. Walk along the main road for 20 minutes to reach the path, cutting across tea plantations before the final climb up. Great views on Ella Rock and Rawana Falls. You can see the Newburgh Tea Estate on the other side (north). Busy during high season. There is a zip line near the bottom of the trail, available for a fee, though it won't save you time. 2 Nine Arch Bridge. Famous beautiful rock bridge east of the town center. 45 minutes walk from the town center (Passara road, same as Little Adam's Peak and Newburgh Estate), then take a left to walk down to the bridge via a paved road and concrete stairs. Alternatively, walk along the train tracks from Ella railway station (but be careful to listen for approaching trains). If unsure, ask the locals, but not tuk tuk drivers, as they will want to take you along an expensive, circuitous route that will take you to near the bridge at track level. Can be a bit crowded with tourists, come early to enjoy the fresh air. Check the time tables to be able to see a train pass while you are there (but there will be around 4 trains passing between 10AM and 1PM). ...
[Do] Have a Massage at one of the local massage places in town. 1 Ella Organic Tea Garden, 5 km south of Ella centre just off Kithalella road, ☏ +94 711344503. 09:00, 11:00, 13:00. Visit one of the few organic tea growers in Sri Lanka. You will get a short tour of the tea plantation, being introduced to several spices and fruits that grow there. After that, learn how the different types of tea are made and taste them all. Rs 3,000 per person. (updated Aug 2024) 2 Newburgh Green Tea Factory. A bit further down Passara Road (after Little Adam's Peak path), and 30 minutes from the town center. Quick visit of the tea factory, including green tea tasting at the end.
Sections ending in ... were shortened; full article: https://en.wikivoyage.org/wiki/Ella
Source: Wikivoyage, community-written (CC BY-SA 4.0); prices, timetables and transport details may be out of date.
Mention only sights named above, say they come from Wikivoyage because OpenStreetMap could not be reached, and do not invent attractions or distances.
```

### get_weather_forecast
```
Weather forecast for Ella Town (Badulla District, Uva Province, Sri Lanka), at lat 6.88, lon 81.05, elevation 1031 m.
Times are local (Asia/Colombo).
Current conditions (2026-11-16 20:15): 20.0 °C, partly cloudy, wind 1.3 km/h, precipitation 0.0 mm.
Daily forecast (3 days):
- 2026-11-16 (Mon): moderate drizzle; min 16.0 °C, max 25.9 °C; chance of precipitation 70%; rainfall 2.4 mm; max wind 8.7 km/h.
- 2026-11-17 (Tue): moderate rain showers; min 16.0 °C, max 26.5 °C; chance of precipitation 92%; rainfall 8.3 mm; max wind 4.7 km/h.
- 2026-11-18 (Wed): moderate drizzle; min 16.1 °C, max 25.5 °C; chance of precipitation 56%; rainfall 2.0 mm; max wind 7.0 km/h.
Source: Open-Meteo live forecast API (open-meteo.com).
```

### estimate_travel_time
```
Road trip from Colombo (Colombo District, Western Province, Sri Lanka) to Ella Town (Badulla District, Uva Province, Sri Lanka):
- Road distance: 203 km (straight line 132 km).
- Driving time: about 3 h 42 min by car without traffic or stops.
This is a free-flow car estimate: real journeys, especially on hill roads or by bus, tuk-tuk or train, usually take longer. Say so.
Source: OSRM routing on OpenStreetMap roads (© OpenStreetMap contributors).
```

### convert_currency
```
3,000.00 LKR = 9.01 USD
Rate used: 1 LKR = 0.003003 USD (updated Mon, 16 Nov 2026 00:02:31 +0000).
Indicative mid-market rate: banks, cards and money changers charge a margin, so the traveller will get somewhat less.
Source: Rates By Exchange Rate API (https://www.exchangerate-api.com).
```

### estimate_travel_time
```
Road trip from Ella Town (Badulla District, Uva Province, Sri Lanka) to Badulla (Uva Province, Sri Lanka):
- Road distance: 21 km (straight line 12 km).
- Driving time: about 28 min by car without traffic or stops.
This is a free-flow car estimate: real journeys, especially on hill roads or by bus, tuk-tuk or train, usually take longer. Say so.
Source: OSRM routing on OpenStreetMap roads (© OpenStreetMap contributors).
```
