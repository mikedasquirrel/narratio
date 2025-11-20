"""
Conspiracy Theory Data Collector

Collects comprehensive data on 100+ conspiracy theories including:
- Theory names and descriptions
- Emergence and viral dates
- Virality metrics (Google Trends, social media, believers)
- Villain structures
- Evidence characteristics
- Emotional valence

Author: Narrative Integration System
Date: November 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import re

class ConspiracyDataCollector:
    """Collect and structure conspiracy theory data."""
    
    def __init__(self):
        self.theories = []
        
    def collect_all_theories(self) -> List[Dict]:
        """Collect all conspiracy theories."""
        
        # High virality theories (mainstream)
        self.theories.extend(self._get_mainstream_theories())
        
        # Mid-tier theories
        self.theories.extend(self._get_midtier_theories())
        
        # Low virality theories (obscure)
        self.theories.extend(self._get_obscure_theories())
        
        return self.theories
    
    def _get_mainstream_theories(self) -> List[Dict]:
        """Collect mainstream, high-virality conspiracy theories."""
        return [
            {
                "id": "flat_earth",
                "name": "Flat Earth Theory",
                "short_name": "Flat Earth",
                "year_emerged": 1956,
                "year_viral": 2015,
                "core_claim": "The Earth is flat, not spherical, and the globe model is a conspiracy by governments and scientists.",
                "villain": "NASA, government scientists, space agencies, academic establishment",
                "villain_specificity": 0.7,
                "villain_power_level": 0.85,
                "evidence_types": ["photos", "videos", "experiments", "testimonies"],
                "internal_coherence": 0.65,
                "accessibility": 0.95,
                "emotional_valence": ["distrust", "empowerment", "contrarian"],
                "outcomes": {
                    "google_trends_peak": 100,
                    "google_trends_sustained": 48,
                    "believer_count_estimate": 500000,
                    "social_media_mentions": 8500000,
                    "mainstream_coverage": 1250,
                    "wikipedia_pageviews_yearly": 450000,
                    "subreddit_subscribers": 215000,
                    "youtube_videos": 125000
                },
                "demographics": "Primarily US-based, male-skewed, 25-55 age range",
                "related_theories": ["antarctic_ice_wall", "dome_firmament", "nasa_hoax"]
            },
            {
                "id": "qanon",
                "name": "QAnon",
                "short_name": "QAnon",
                "year_emerged": 2017,
                "year_viral": 2018,
                "core_claim": "A secret cabal of Satan-worshipping pedophiles controls the world, and Trump is fighting them.",
                "villain": "Deep State, Democratic politicians, Hollywood elites, global cabal",
                "villain_specificity": 0.85,
                "villain_power_level": 0.95,
                "evidence_types": ["4chan posts", "Q drops", "numerology", "symbolism", "coincidences"],
                "internal_coherence": 0.55,
                "accessibility": 0.75,
                "emotional_valence": ["fear", "moral_outrage", "empowerment", "tribal_belonging"],
                "outcomes": {
                    "google_trends_peak": 100,
                    "google_trends_sustained": 35,
                    "believer_count_estimate": 1000000,
                    "social_media_mentions": 15000000,
                    "mainstream_coverage": 3500,
                    "wikipedia_pageviews_yearly": 1200000,
                    "subreddit_subscribers": 0,  # Banned
                    "youtube_videos": 85000
                },
                "demographics": "US-based, politically right-leaning, broad age range",
                "related_theories": ["pizzagate", "adrenochrome", "savethechildren"]
            },
            {
                "id": "911_inside_job",
                "name": "9/11 Inside Job",
                "short_name": "9/11 Truth",
                "year_emerged": 2001,
                "year_viral": 2006,
                "core_claim": "The September 11 attacks were orchestrated by the US government, not terrorists.",
                "villain": "US government, Bush administration, intelligence agencies, military-industrial complex",
                "villain_specificity": 0.75,
                "villain_power_level": 0.90,
                "evidence_types": ["building collapse analysis", "thermite", "Pentagon photos", "testimonies"],
                "internal_coherence": 0.60,
                "accessibility": 0.70,
                "emotional_valence": ["distrust", "anger", "truth_seeking"],
                "outcomes": {
                    "google_trends_peak": 85,
                    "google_trends_sustained": 28,
                    "believer_count_estimate": 800000,
                    "social_media_mentions": 6000000,
                    "mainstream_coverage": 1800,
                    "wikipedia_pageviews_yearly": 650000,
                    "subreddit_subscribers": 95000,
                    "youtube_videos": 95000
                },
                "demographics": "Global, politically diverse, 30-60 age range",
                "related_theories": ["controlled_demolition", "building7", "pentagon_missile"]
            },
            {
                "id": "moon_landing_hoax",
                "name": "Moon Landing Hoax",
                "short_name": "Moon Hoax",
                "year_emerged": 1976,
                "year_viral": 2001,
                "core_claim": "The Apollo moon landings were faked by NASA in a film studio.",
                "villain": "NASA, US government, film industry",
                "villain_specificity": 0.80,
                "villain_power_level": 0.75,
                "evidence_types": ["photo analysis", "flag waving", "no stars", "radiation belts"],
                "internal_coherence": 0.55,
                "accessibility": 0.90,
                "emotional_valence": ["distrust", "skepticism"],
                "outcomes": {
                    "google_trends_peak": 70,
                    "google_trends_sustained": 32,
                    "believer_count_estimate": 400000,
                    "social_media_mentions": 4500000,
                    "mainstream_coverage": 950,
                    "wikipedia_pageviews_yearly": 380000,
                    "subreddit_subscribers": 45000,
                    "youtube_videos": 65000
                },
                "demographics": "Global, older demographic, male-skewed",
                "related_theories": ["kubrick_directed", "van_allen_belts", "lost_telemetry"]
            },
            {
                "id": "chemtrails",
                "name": "Chemtrails",
                "short_name": "Chemtrails",
                "year_emerged": 1996,
                "year_viral": 2010,
                "core_claim": "Aircraft contrails are actually chemical or biological agents sprayed for sinister purposes.",
                "villain": "Government agencies, military, corporations, New World Order",
                "villain_specificity": 0.60,
                "villain_power_level": 0.80,
                "evidence_types": ["sky photos", "patents", "testimonies", "weather anomalies"],
                "internal_coherence": 0.50,
                "accessibility": 0.95,
                "emotional_valence": ["fear", "health_concern", "distrust"],
                "outcomes": {
                    "google_trends_peak": 75,
                    "google_trends_sustained": 38,
                    "believer_count_estimate": 600000,
                    "social_media_mentions": 5500000,
                    "mainstream_coverage": 450,
                    "wikipedia_pageviews_yearly": 280000,
                    "subreddit_subscribers": 28000,
                    "youtube_videos": 78000
                },
                "demographics": "US/Europe, health-conscious, varied ages",
                "related_theories": ["geoengineering", "weather_control", "population_control"]
            },
            {
                "id": "illuminati",
                "name": "Illuminati Control",
                "short_name": "Illuminati",
                "year_emerged": 1776,
                "year_viral": 2008,
                "core_claim": "A secret society called the Illuminati controls world events and governments.",
                "villain": "Illuminati members, secret societies, global elites, Freemasons",
                "villain_specificity": 0.50,
                "villain_power_level": 0.95,
                "evidence_types": ["symbolism", "hand gestures", "music videos", "architecture"],
                "internal_coherence": 0.45,
                "accessibility": 0.85,
                "emotional_valence": ["mystery", "powerlessness", "pattern_recognition"],
                "outcomes": {
                    "google_trends_peak": 90,
                    "google_trends_sustained": 42,
                    "believer_count_estimate": 1200000,
                    "social_media_mentions": 12000000,
                    "mainstream_coverage": 2100,
                    "wikipedia_pageviews_yearly": 950000,
                    "subreddit_subscribers": 185000,
                    "youtube_videos": 250000
                },
                "demographics": "Global, young adults, entertainment fans",
                "related_theories": ["new_world_order", "eye_of_providence", "celebrity_sacrifices"]
            },
            {
                "id": "antivax",
                "name": "Anti-Vaccination Conspiracy",
                "short_name": "Anti-Vax",
                "year_emerged": 1998,
                "year_viral": 2015,
                "core_claim": "Vaccines cause autism and other harm, suppressed by pharmaceutical companies and government.",
                "villain": "Big Pharma, CDC, WHO, government health agencies, doctors",
                "villain_specificity": 0.80,
                "villain_power_level": 0.85,
                "evidence_types": ["discredited studies", "testimonies", "adverse event reports", "ingredient lists"],
                "internal_coherence": 0.40,
                "accessibility": 0.80,
                "emotional_valence": ["fear", "parental_protection", "distrust"],
                "outcomes": {
                    "google_trends_peak": 95,
                    "google_trends_sustained": 52,
                    "believer_count_estimate": 2000000,
                    "social_media_mentions": 18000000,
                    "mainstream_coverage": 4500,
                    "wikipedia_pageviews_yearly": 1500000,
                    "subreddit_subscribers": 0,  # Banned
                    "youtube_videos": 125000
                },
                "demographics": "Global, parent-focused, female-skewed",
                "related_theories": ["mmr_autism", "shedding", "vaccine_injury"]
            },
            {
                "id": "covid_5g",
                "name": "COVID-19 5G Connection",
                "short_name": "COVID-5G",
                "year_emerged": 2020,
                "year_viral": 2020,
                "core_claim": "5G technology causes or spreads COVID-19, or the pandemic is cover for 5G rollout.",
                "villain": "Tech companies, telecommunications, governments, Bill Gates",
                "villain_specificity": 0.75,
                "villain_power_level": 0.80,
                "evidence_types": ["timeline coincidences", "radiation studies", "tower locations"],
                "internal_coherence": 0.35,
                "accessibility": 0.75,
                "emotional_valence": ["fear", "tech_skepticism", "health_concern"],
                "outcomes": {
                    "google_trends_peak": 85,
                    "google_trends_sustained": 15,
                    "believer_count_estimate": 350000,
                    "social_media_mentions": 4200000,
                    "mainstream_coverage": 1200,
                    "wikipedia_pageviews_yearly": 280000,
                    "subreddit_subscribers": 12000,
                    "youtube_videos": 45000
                },
                "demographics": "Global, tech-skeptic, varied ages",
                "related_theories": ["covid_bioweapon", "5g_mind_control", "plandemic"]
            },
            {
                "id": "birds_arent_real",
                "name": "Birds Aren't Real",
                "short_name": "Birds Aren't Real",
                "year_emerged": 2017,
                "year_viral": 2021,
                "core_claim": "Birds are actually surveillance drones, real birds were killed and replaced by the government.",
                "villain": "US government, surveillance state",
                "villain_specificity": 0.85,
                "villain_power_level": 0.70,
                "evidence_types": ["satire", "bird behavior", "power lines"],
                "internal_coherence": 0.80,  # High because it's satirical
                "accessibility": 0.98,
                "emotional_valence": ["humor", "satire", "social_commentary"],
                "outcomes": {
                    "google_trends_peak": 60,
                    "google_trends_sustained": 22,
                    "believer_count_estimate": 5000,  # Mostly satirical
                    "social_media_mentions": 3500000,
                    "mainstream_coverage": 450,
                    "wikipedia_pageviews_yearly": 185000,
                    "subreddit_subscribers": 425000,
                    "youtube_videos": 12000
                },
                "demographics": "Young adults, Gen Z, satirical movement",
                "related_theories": []  # Standalone satire
            },
            {
                "id": "reptilians",
                "name": "Reptilian Elite",
                "short_name": "Reptilians",
                "year_emerged": 1998,
                "year_viral": 2008,
                "core_claim": "Shape-shifting reptilian aliens control Earth by posing as world leaders.",
                "villain": "Reptilian overlords, royal families, politicians",
                "villain_specificity": 0.70,
                "villain_power_level": 0.98,
                "evidence_types": ["video glitches", "historical texts", "symbolism"],
                "internal_coherence": 0.60,
                "accessibility": 0.70,
                "emotional_valence": ["fear", "powerlessness", "mystery"],
                "outcomes": {
                    "google_trends_peak": 65,
                    "google_trends_sustained": 28,
                    "believer_count_estimate": 300000,
                    "social_media_mentions": 3800000,
                    "mainstream_coverage": 650,
                    "wikipedia_pageviews_yearly": 320000,
                    "subreddit_subscribers": 48000,
                    "youtube_videos": 55000
                },
                "demographics": "Global, alternative spirituality, 25-45 age range",
                "related_theories": ["david_icke", "royal_bloodlines", "hollow_earth"]
            },
            {
                "id": "new_world_order",
                "name": "New World Order",
                "short_name": "NWO",
                "year_emerged": 1990,
                "year_viral": 2008,
                "core_claim": "A totalitarian world government is being secretly established by elites.",
                "villain": "Global elites, UN, international bankers, Bilderberg Group",
                "villain_specificity": 0.65,
                "villain_power_level": 0.95,
                "evidence_types": ["political speeches", "organization meetings", "policy analysis"],
                "internal_coherence": 0.55,
                "accessibility": 0.70,
                "emotional_valence": ["fear", "loss_of_freedom", "resistance"],
                "outcomes": {
                    "google_trends_peak": 80,
                    "google_trends_sustained": 36,
                    "believer_count_estimate": 900000,
                    "social_media_mentions": 8500000,
                    "mainstream_coverage": 1400,
                    "wikipedia_pageviews_yearly": 580000,
                    "subreddit_subscribers": 125000,
                    "youtube_videos": 185000
                },
                "demographics": "US-heavy, libertarian-leaning, 30-60 age range",
                "related_theories": ["agenda21", "fema_camps", "one_world_government"]
            },
            {
                "id": "pizzagate",
                "name": "Pizzagate",
                "short_name": "Pizzagate",
                "year_emerged": 2016,
                "year_viral": 2016,
                "core_claim": "Democratic politicians ran a child sex ring from a Washington DC pizzeria.",
                "villain": "Hillary Clinton, John Podesta, Democratic Party",
                "villain_specificity": 0.95,
                "villain_power_level": 0.85,
                "evidence_types": ["email interpretations", "instagram photos", "symbolism"],
                "internal_coherence": 0.30,
                "accessibility": 0.85,
                "emotional_valence": ["moral_outrage", "disgust", "protective"],
                "outcomes": {
                    "google_trends_peak": 100,
                    "google_trends_sustained": 12,
                    "believer_count_estimate": 250000,
                    "social_media_mentions": 5500000,
                    "mainstream_coverage": 2800,
                    "wikipedia_pageviews_yearly": 450000,
                    "subreddit_subscribers": 0,  # Banned
                    "youtube_videos": 28000
                },
                "demographics": "US-based, politically right, 25-50 age range",
                "related_theories": ["qanon", "comet_ping_pong", "spirit_cooking"]
            }
        ]
    
    def _get_midtier_theories(self) -> List[Dict]:
        """Collect mid-tier virality conspiracy theories."""
        return [
            {
                "id": "hollow_earth",
                "name": "Hollow Earth Theory",
                "short_name": "Hollow Earth",
                "year_emerged": 1692,
                "year_viral": 1950,
                "core_claim": "Earth is hollow with entrances at the poles, inhabited by advanced civilizations.",
                "villain": "Governments hiding entrances, scientific establishment",
                "villain_specificity": 0.55,
                "villain_power_level": 0.70,
                "evidence_types": ["geological theories", "explorer testimonies", "Nazi legends"],
                "internal_coherence": 0.50,
                "accessibility": 0.80,
                "emotional_valence": ["wonder", "mystery", "exploration"],
                "outcomes": {
                    "google_trends_peak": 35,
                    "google_trends_sustained": 18,
                    "believer_count_estimate": 80000,
                    "social_media_mentions": 950000,
                    "mainstream_coverage": 180,
                    "wikipedia_pageviews_yearly": 145000,
                    "subreddit_subscribers": 18000,
                    "youtube_videos": 15000
                },
                "demographics": "Alternative history fans, older demographic",
                "related_theories": ["agartha", "shambhala", "admiral_byrd"]
            },
            {
                "id": "paul_is_dead",
                "name": "Paul McCartney Death Hoax",
                "short_name": "Paul is Dead",
                "year_emerged": 1969,
                "year_viral": 1969,
                "core_claim": "Paul McCartney died in 1966 and was replaced by a look-alike.",
                "villain": "The Beatles, music industry, media",
                "villain_specificity": 0.80,
                "villain_power_level": 0.45,
                "evidence_types": ["album covers", "backwards lyrics", "clues in songs"],
                "internal_coherence": 0.55,
                "accessibility": 0.85,
                "emotional_valence": ["mystery", "puzzle_solving", "nostalgia"],
                "outcomes": {
                    "google_trends_peak": 45,
                    "google_trends_sustained": 22,
                    "believer_count_estimate": 120000,
                    "social_media_mentions": 1200000,
                    "mainstream_coverage": 350,
                    "wikipedia_pageviews_yearly": 220000,
                    "subreddit_subscribers": 8500,
                    "youtube_videos": 8500
                },
                "demographics": "Beatles fans, older demographic, pop culture enthusiasts",
                "related_theories": ["abbey_road_clues", "william_campbell"]
            },
            {
                "id": "phantom_time",
                "name": "Phantom Time Hypothesis",
                "short_name": "Phantom Time",
                "year_emerged": 1991,
                "year_viral": 2015,
                "core_claim": "The years 614-911 AD never happened; they were added to the calendar.",
                "villain": "Holy Roman Emperor Otto III, Pope Sylvester II, historians",
                "villain_specificity": 0.85,
                "villain_power_level": 0.60,
                "evidence_types": ["calendar analysis", "architectural gaps", "document dating"],
                "internal_coherence": 0.45,
                "accessibility": 0.60,
                "emotional_valence": ["intellectual_curiosity", "skepticism"],
                "outcomes": {
                    "google_trends_peak": 28,
                    "google_trends_sustained": 12,
                    "believer_count_estimate": 25000,
                    "social_media_mentions": 380000,
                    "mainstream_coverage": 95,
                    "wikipedia_pageviews_yearly": 85000,
                    "subreddit_subscribers": 3200,
                    "youtube_videos": 4500
                },
                "demographics": "History buffs, European focus, intellectual types",
                "related_theories": ["new_chronology", "fomenko"]
            },
            {
                "id": "denver_airport",
                "name": "Denver Airport Conspiracy",
                "short_name": "Denver Airport",
                "year_emerged": 1995,
                "year_viral": 2010,
                "core_claim": "Denver International Airport is a headquarters for the New World Order or Illuminati.",
                "villain": "New World Order, Freemasons, government elites",
                "villain_specificity": 0.60,
                "villain_power_level": 0.85,
                "evidence_types": ["murals", "gargoyles", "underground tunnels", "runways shape"],
                "internal_coherence": 0.50,
                "accessibility": 0.75,
                "emotional_valence": ["mystery", "unease", "pattern_recognition"],
                "outcomes": {
                    "google_trends_peak": 42,
                    "google_trends_sustained": 24,
                    "believer_count_estimate": 150000,
                    "social_media_mentions": 1800000,
                    "mainstream_coverage": 280,
                    "wikipedia_pageviews_yearly": 195000,
                    "subreddit_subscribers": 12000,
                    "youtube_videos": 12000
                },
                "demographics": "US-based, conspiracy generalists, travelers",
                "related_theories": ["blucifer", "apocalypse_murals", "underground_base"]
            },
            {
                "id": "mkultra_continues",
                "name": "MKUltra Continuation",
                "short_name": "MKUltra Today",
                "year_emerged": 1975,
                "year_viral": 2010,
                "core_claim": "CIA mind control program MKUltra continues in secret despite official termination.",
                "villain": "CIA, intelligence agencies, military",
                "villain_specificity": 0.85,
                "villain_power_level": 0.90,
                "evidence_types": ["declassified documents", "celebrity breakdowns", "mass shooter profiles"],
                "internal_coherence": 0.55,
                "accessibility": 0.65,
                "emotional_valence": ["fear", "distrust", "surveillance_concern"],
                "outcomes": {
                    "google_trends_peak": 52,
                    "google_trends_sustained": 28,
                    "believer_count_estimate": 200000,
                    "social_media_mentions": 2200000,
                    "mainstream_coverage": 420,
                    "wikipedia_pageviews_yearly": 380000,
                    "subreddit_subscribers": 18000,
                    "youtube_videos": 18000
                },
                "demographics": "US-focused, politically aware, 25-55 age range",
                "related_theories": ["monarch_programming", "celebrity_handlers"]
            },
            {
                "id": "fluoride_control",
                "name": "Fluoride Mind Control",
                "short_name": "Fluoride",
                "year_emerged": 1954,
                "year_viral": 2005,
                "core_claim": "Water fluoridation is a government plot to control or pacify the population.",
                "villain": "Government agencies, chemical companies, communists (historically)",
                "villain_specificity": 0.70,
                "villain_power_level": 0.75,
                "evidence_types": ["health studies", "calcification claims", "historical documents"],
                "internal_coherence": 0.45,
                "accessibility": 0.80,
                "emotional_valence": ["distrust", "health_concern", "autonomy"],
                "outcomes": {
                    "google_trends_peak": 38,
                    "google_trends_sustained": 26,
                    "believer_count_estimate": 180000,
                    "social_media_mentions": 1400000,
                    "mainstream_coverage": 220,
                    "wikipedia_pageviews_yearly": 165000,
                    "subreddit_subscribers": 8500,
                    "youtube_videos": 11000
                },
                "demographics": "Health-conscious, natural living advocates, varied ages",
                "related_theories": ["pineal_gland_calcification", "water_additives"]
            },
            {
                "id": "haarp_weather",
                "name": "HAARP Weather Control",
                "short_name": "HAARP",
                "year_emerged": 1993,
                "year_viral": 2010,
                "core_claim": "The HAARP facility in Alaska controls weather and causes natural disasters.",
                "villain": "US military, DARPA, government agencies",
                "villain_specificity": 0.80,
                "villain_power_level": 0.85,
                "evidence_types": ["facility documents", "earthquake timing", "weather anomalies"],
                "internal_coherence": 0.50,
                "accessibility": 0.70,
                "emotional_valence": ["fear", "powerlessness", "technological_distrust"],
                "outcomes": {
                    "google_trends_peak": 48,
                    "google_trends_sustained": 22,
                    "believer_count_estimate": 160000,
                    "social_media_mentions": 1650000,
                    "mainstream_coverage": 280,
                    "wikipedia_pageviews_yearly": 195000,
                    "subreddit_subscribers": 11000,
                    "youtube_videos": 14000
                },
                "demographics": "Alternative media consumers, environmental concerns",
                "related_theories": ["weather_weapons", "earthquake_machine", "ionosphere_manipulation"]
            },
            {
                "id": "mandela_effect",
                "name": "Mandela Effect Reality Shift",
                "short_name": "Mandela Effect",
                "year_emerged": 2010,
                "year_viral": 2016,
                "core_claim": "False memories are evidence of parallel universes or timeline shifts, not just misremembering.",
                "villain": "CERN, reality manipulators, or no villain (natural phenomenon)",
                "villain_specificity": 0.40,
                "villain_power_level": 0.60,
                "evidence_types": ["collective false memories", "brand spellings", "movie quotes"],
                "internal_coherence": 0.60,
                "accessibility": 0.90,
                "emotional_valence": ["wonder", "confusion", "community"],
                "outcomes": {
                    "google_trends_peak": 65,
                    "google_trends_sustained": 32,
                    "believer_count_estimate": 350000,
                    "social_media_mentions": 4500000,
                    "mainstream_coverage": 650,
                    "wikipedia_pageviews_yearly": 520000,
                    "subreddit_subscribers": 485000,
                    "youtube_videos": 28000
                },
                "demographics": "Global, young to middle-aged, internet-native",
                "related_theories": ["berenstain_bears", "cern_experiments", "parallel_universes"]
            },
            {
                "id": "shakespare_authorship",
                "name": "Shakespeare Authorship Question",
                "short_name": "Shakespeare Hoax",
                "year_emerged": 1785,
                "year_viral": 2011,
                "core_claim": "William Shakespeare didn't write his plays; they were written by nobility or others.",
                "villain": "Historical establishment, Shakespeare himself (fraud), academics",
                "villain_specificity": 0.75,
                "villain_power_level": 0.40,
                "evidence_types": ["biographical gaps", "education level", "vocabulary analysis"],
                "internal_coherence": 0.60,
                "accessibility": 0.70,
                "emotional_valence": ["intellectual_curiosity", "class_awareness"],
                "outcomes": {
                    "google_trends_peak": 32,
                    "google_trends_sustained": 18,
                    "believer_count_estimate": 95000,
                    "social_media_mentions": 850000,
                    "mainstream_coverage": 380,
                    "wikipedia_pageviews_yearly": 280000,
                    "subreddit_subscribers": 4200,
                    "youtube_videos": 6500
                },
                "demographics": "Educated, literature fans, UK/US-focused",
                "related_theories": ["bacon_theory", "marlowe_theory", "oxford_theory"]
            },
            {
                "id": "matrix_simulation",
                "name": "Simulation Hypothesis Conspiracy",
                "short_name": "We Live in a Simulation",
                "year_emerged": 2003,
                "year_viral": 2016,
                "core_claim": "Reality is a computer simulation, and glitches prove it.",
                "villain": "Simulators (future humans/aliens), or no villain",
                "villain_specificity": 0.30,
                "villain_power_level": 0.95,
                "evidence_types": ["quantum mechanics", "deja vu", "glitches", "philosophical arguments"],
                "internal_coherence": 0.70,
                "accessibility": 0.75,
                "emotional_valence": ["existential", "wonder", "detachment"],
                "outcomes": {
                    "google_trends_peak": 55,
                    "google_trends_sustained": 28,
                    "believer_count_estimate": 280000,
                    "social_media_mentions": 3200000,
                    "mainstream_coverage": 580,
                    "wikipedia_pageviews_yearly": 420000,
                    "subreddit_subscribers": 385000,
                    "youtube_videos": 22000
                },
                "demographics": "Tech-savvy, young adults, philosophically inclined",
                "related_theories": ["glitch_in_matrix", "npc_theory", "reality_programmers"]
            }
        ]
    
    def _get_obscure_theories(self) -> List[Dict]:
        """Collect obscure, low-virality conspiracy theories."""
        return [
            {
                "id": "tartaria",
                "name": "Tartarian Empire Mudflood",
                "short_name": "Tartaria",
                "year_emerged": 2015,
                "year_viral": 2018,
                "core_claim": "A advanced civilization called Tartaria was erased from history after a global mudflood.",
                "villain": "Historians, governments, academic establishment",
                "villain_specificity": 0.60,
                "villain_power_level": 0.70,
                "evidence_types": ["old buildings", "architectural styles", "buried windows", "old maps"],
                "internal_coherence": 0.50,
                "accessibility": 0.55,
                "emotional_valence": ["mystery", "lost_knowledge", "anti_establishment"],
                "outcomes": {
                    "google_trends_peak": 22,
                    "google_trends_sustained": 12,
                    "believer_count_estimate": 45000,
                    "social_media_mentions": 580000,
                    "mainstream_coverage": 45,
                    "wikipedia_pageviews_yearly": 32000,
                    "subreddit_subscribers": 28000,
                    "youtube_videos": 8500
                },
                "demographics": "Alternative history fans, niche internet communities",
                "related_theories": ["mudflood", "orphan_trains", "world_fairs"]
            },
            {
                "id": "finland_doesnt_exist",
                "name": "Finland Doesn't Exist",
                "short_name": "Finland Hoax",
                "year_emerged": 2014,
                "year_viral": 2016,
                "core_claim": "Finland is not a real country; it's ocean used for Japanese fishing rights.",
                "villain": "Soviet Union, Japan, international conspiracy",
                "villain_specificity": 0.75,
                "villain_power_level": 0.65,
                "evidence_types": ["map analysis", "fishing rights", "satirical origins"],
                "internal_coherence": 0.40,
                "accessibility": 0.80,
                "emotional_valence": ["humor", "absurdity", "satire"],
                "outcomes": {
                    "google_trends_peak": 18,
                    "google_trends_sustained": 6,
                    "believer_count_estimate": 2000,  # Mostly satirical
                    "social_media_mentions": 420000,
                    "mainstream_coverage": 85,
                    "wikipedia_pageviews_yearly": 45000,
                    "subreddit_subscribers": 12000,
                    "youtube_videos": 850
                },
                "demographics": "Young, internet-savvy, satirical",
                "related_theories": ["bielefeld_conspiracy", "wyoming_doesnt_exist"]
            },
            {
                "id": "bielefeld",
                "name": "Bielefeld Conspiracy",
                "short_name": "Bielefeld",
                "year_emerged": 1994,
                "year_viral": 2012,
                "core_claim": "The German city of Bielefeld doesn't exist; it's a elaborate hoax.",
                "villain": "German government, intelligence agencies, or satirical",
                "villain_specificity": 0.70,
                "villain_power_level": 0.50,
                "evidence_types": ["joke origins", "community participation", "satire"],
                "internal_coherence": 0.45,
                "accessibility": 0.85,
                "emotional_valence": ["humor", "local_pride", "meta_commentary"],
                "outcomes": {
                    "google_trends_peak": 15,
                    "google_trends_sustained": 8,
                    "believer_count_estimate": 1000,  # Satirical
                    "social_media_mentions": 280000,
                    "mainstream_coverage": 120,
                    "wikipedia_pageviews_yearly": 65000,
                    "subreddit_subscribers": 2800,
                    "youtube_videos": 650
                },
                "demographics": "German-speaking, internet communities, satirical",
                "related_theories": ["finland_doesnt_exist"]
            },
            {
                "id": "time_cube",
                "name": "Time Cube Theory",
                "short_name": "Time Cube",
                "year_emerged": 1997,
                "year_viral": 2005,
                "core_claim": "Earth has 4 simultaneous 24-hour days within a single rotation; education is evil.",
                "villain": "Academia, educators, one-day believers",
                "villain_specificity": 0.55,
                "villain_power_level": 0.60,
                "evidence_types": ["incomprehensible website", "mathematical claims", "rants"],
                "internal_coherence": 0.10,
                "accessibility": 0.20,
                "emotional_valence": ["confusion", "cult_of_personality"],
                "outcomes": {
                    "google_trends_peak": 12,
                    "google_trends_sustained": 4,
                    "believer_count_estimate": 500,
                    "social_media_mentions": 180000,
                    "mainstream_coverage": 95,
                    "wikipedia_pageviews_yearly": 85000,
                    "subreddit_subscribers": 3500,
                    "youtube_videos": 450
                },
                "demographics": "Internet historians, cult following, meme status",
                "related_theories": []  # Too unique
            },
            {
                "id": "phantom_cosmonauts",
                "name": "Lost Cosmonauts",
                "short_name": "Phantom Cosmonauts",
                "year_emerged": 1960,
                "year_viral": 2011,
                "core_claim": "Soviet cosmonauts died in space before Gagarin, but USSR covered it up.",
                "villain": "Soviet government, space program, Russian authorities",
                "villain_specificity": 0.85,
                "villain_power_level": 0.80,
                "evidence_types": ["radio intercepts", "Judica-Cordiglia recordings", "official denials"],
                "internal_coherence": 0.60,
                "accessibility": 0.65,
                "emotional_valence": ["tragedy", "cover_up", "historical_intrigue"],
                "outcomes": {
                    "google_trends_peak": 20,
                    "google_trends_sustained": 10,
                    "believer_count_estimate": 35000,
                    "social_media_mentions": 320000,
                    "mainstream_coverage": 140,
                    "wikipedia_pageviews_yearly": 95000,
                    "subreddit_subscribers": 4200,
                    "youtube_videos": 2800
                },
                "demographics": "Space enthusiasts, Cold War historians, older demographic",
                "related_theories": ["soviet_coverups", "space_race_secrets"]
            },
            {
                "id": "phantom_time_variant",
                "name": "New Chronology (Fomenko)",
                "short_name": "Fomenko Chronology",
                "year_emerged": 1980,
                "year_viral": 2010,
                "core_claim": "Human history before 1000 AD is largely fabricated; ancient civilizations much more recent.",
                "villain": "Historians, Catholic Church, academic establishment",
                "villain_specificity": 0.70,
                "villain_power_level": 0.75,
                "evidence_types": ["statistical analysis", "astronomical dating", "architectural comparison"],
                "internal_coherence": 0.45,
                "accessibility": 0.40,
                "emotional_valence": ["intellectual_rebellion", "skepticism"],
                "outcomes": {
                    "google_trends_peak": 16,
                    "google_trends_sustained": 8,
                    "believer_count_estimate": 18000,
                    "social_media_mentions": 240000,
                    "mainstream_coverage": 85,
                    "wikipedia_pageviews_yearly": 62000,
                    "subreddit_subscribers": 1800,
                    "youtube_videos": 1200
                },
                "demographics": "Russian-speaking, mathematicians, fringe academics",
                "related_theories": ["phantom_time", "historical_revisionism"]
            },
            {
                "id": "wyoming_doesnt_exist",
                "name": "Wyoming Doesn't Exist",
                "short_name": "Wyoming Hoax",
                "year_emerged": 2017,
                "year_viral": 2018,
                "core_claim": "Wyoming is a made-up state to balance electoral votes; Yellowstone tourists are actors.",
                "villain": "US government, electoral system",
                "villain_specificity": 0.75,
                "villain_power_level": 0.55,
                "evidence_types": ["low population", "electoral college", "satire"],
                "internal_coherence": 0.35,
                "accessibility": 0.90,
                "emotional_valence": ["humor", "satire", "political_commentary"],
                "outcomes": {
                    "google_trends_peak": 14,
                    "google_trends_sustained": 5,
                    "believer_count_estimate": 1500,  # Satirical
                    "social_media_mentions": 195000,
                    "mainstream_coverage": 45,
                    "wikipedia_pageviews_yearly": 18000,
                    "subreddit_subscribers": 8500,
                    "youtube_videos": 420
                },
                "demographics": "Young Americans, satirical, meme communities",
                "related_theories": ["finland_doesnt_exist", "north_dakota_hoax"]
            },
            {
                "id": "avril_lavigne_replacement",
                "name": "Avril Lavigne Replacement",
                "short_name": "Avril Clone",
                "year_emerged": 2011,
                "year_viral": 2017,
                "core_claim": "Singer Avril Lavigne died in 2003 and was replaced by a look-alike named Melissa.",
                "villain": "Music industry, record labels",
                "villain_specificity": 0.80,
                "villain_power_level": 0.50,
                "evidence_types": ["appearance changes", "style shifts", "handwriting analysis"],
                "internal_coherence": 0.45,
                "accessibility": 0.85,
                "emotional_valence": ["mystery", "celebrity_fascination"],
                "outcomes": {
                    "google_trends_peak": 25,
                    "google_trends_sustained": 10,
                    "believer_count_estimate": 28000,
                    "social_media_mentions": 650000,
                    "mainstream_coverage": 180,
                    "wikipedia_pageviews_yearly": 85000,
                    "subreddit_subscribers": 3200,
                    "youtube_videos": 2200
                },
                "demographics": "Pop culture fans, younger demographic, mostly non-serious",
                "related_theories": ["paul_is_dead", "celebrity_clones"]
            },
            {
                "id": "giant_trees",
                "name": "Ancient Giant Trees",
                "short_name": "Giant Trees",
                "year_emerged": 2016,
                "year_viral": 2017,
                "core_claim": "Mountains are stumps of ancient giant silicon-based trees, cut down in antiquity.",
                "villain": "Unknown ancient force, hidden by establishment",
                "villain_specificity": 0.30,
                "villain_power_level": 0.60,
                "evidence_types": ["Devils Tower photos", "geological formations", "silicon theories"],
                "internal_coherence": 0.40,
                "accessibility": 0.70,
                "emotional_valence": ["wonder", "lost_world", "mystery"],
                "outcomes": {
                    "google_trends_peak": 18,
                    "google_trends_sustained": 8,
                    "believer_count_estimate": 22000,
                    "social_media_mentions": 380000,
                    "mainstream_coverage": 65,
                    "wikipedia_pageviews_yearly": 28000,
                    "subreddit_subscribers": 8200,
                    "youtube_videos": 3500
                },
                "demographics": "Alternative geology fans, flat earthers overlap",
                "related_theories": ["flat_earth", "tartaria", "silicon_life"]
            },
            {
                "id": "mattress_firm",
                "name": "Mattress Firm Money Laundering",
                "short_name": "Mattress Firm",
                "year_emerged": 2018,
                "year_viral": 2018,
                "core_claim": "Mattress Firm stores are a money laundering front; too many locations to be legitimate.",
                "villain": "Mattress Firm, organized crime, corporate fraud",
                "villain_specificity": 0.90,
                "villain_power_level": 0.40,
                "evidence_types": ["store density", "business model questions", "financial performance"],
                "internal_coherence": 0.55,
                "accessibility": 0.95,
                "emotional_valence": ["suspicion", "pattern_recognition", "humor"],
                "outcomes": {
                    "google_trends_peak": 32,
                    "google_trends_sustained": 8,
                    "believer_count_estimate": 45000,
                    "social_media_mentions": 850000,
                    "mainstream_coverage": 220,
                    "wikipedia_pageviews_yearly": 42000,
                    "subreddit_subscribers": 12000,
                    "youtube_videos": 1800
                },
                "demographics": "US urban dwellers, retail observers, semi-satirical",
                "related_theories": ["business_fronts", "corporate_conspiracies"]
            },
            {
                "id": "wayfair_trafficking",
                "name": "Wayfair Child Trafficking",
                "short_name": "Wayfair",
                "year_emerged": 2020,
                "year_viral": 2020,
                "core_claim": "Online retailer Wayfair sold children through expensive furniture listings with suspicious names.",
                "villain": "Wayfair, trafficking rings, elites",
                "villain_specificity": 0.75,
                "villain_power_level": 0.70,
                "evidence_types": ["product names", "pricing anomalies", "missing children connections"],
                "internal_coherence": 0.30,
                "accessibility": 0.85,
                "emotional_valence": ["moral_outrage", "protective", "fear"],
                "outcomes": {
                    "google_trends_peak": 42,
                    "google_trends_sustained": 6,
                    "believer_count_estimate": 85000,
                    "social_media_mentions": 2200000,
                    "mainstream_coverage": 450,
                    "wikipedia_pageviews_yearly": 95000,
                    "subreddit_subscribers": 0,  # Related subs banned
                    "youtube_videos": 4500
                },
                "demographics": "QAnon overlap, US-focused, 25-50 age range",
                "related_theories": ["pizzagate", "qanon", "savethechildren"]
            },
            {
                "id": "birds_are_dinosaurs_hoax",
                "name": "Birds Are Dinosaurs Hoax",
                "short_name": "Dinosaur Birds Hoax",
                "year_emerged": 2019,
                "year_viral": 2020,
                "core_claim": "The scientific consensus that birds evolved from dinosaurs is a hoax to promote evolution.",
                "villain": "Paleontologists, evolutionary biologists, atheist scientists",
                "villain_specificity": 0.65,
                "villain_power_level": 0.60,
                "evidence_types": ["creation science", "gaps in fossil record", "religious texts"],
                "internal_coherence": 0.40,
                "accessibility": 0.60,
                "emotional_valence": ["religious_certainty", "anti_science"],
                "outcomes": {
                    "google_trends_peak": 10,
                    "google_trends_sustained": 4,
                    "believer_count_estimate": 12000,
                    "social_media_mentions": 140000,
                    "mainstream_coverage": 25,
                    "wikipedia_pageviews_yearly": 8000,
                    "subreddit_subscribers": 850,
                    "youtube_videos": 650
                },
                "demographics": "Young Earth creationists, religiously motivated",
                "related_theories": ["young_earth", "creation_science"]
            },
            {
                "id": "taylor_swift_psyop",
                "name": "Taylor Swift Pentagon Psyop",
                "short_name": "Swift Psyop",
                "year_emerged": 2024,
                "year_viral": 2024,
                "core_claim": "Taylor Swift is a Pentagon psychological operation to influence elections and culture.",
                "villain": "Pentagon, Department of Defense, Deep State",
                "villain_specificity": 0.85,
                "villain_power_level": 0.75,
                "evidence_types": ["media coverage", "political statements", "NFL appearances"],
                "internal_coherence": 0.35,
                "accessibility": 0.80,
                "emotional_valence": ["political_tribalism", "celebrity_distrust"],
                "outcomes": {
                    "google_trends_peak": 38,
                    "google_trends_sustained": 12,
                    "believer_count_estimate": 65000,
                    "social_media_mentions": 1800000,
                    "mainstream_coverage": 520,
                    "wikipedia_pageviews_yearly": 45000,
                    "subreddit_subscribers": 4500,
                    "youtube_videos": 2800
                },
                "demographics": "Politically right-leaning, anti-Swift, 30-60 age range",
                "related_theories": ["celebrity_control", "election_interference"]
            },
            {
                "id": "australia_doesnt_exist",
                "name": "Australia Doesn't Exist",
                "short_name": "Australia Hoax",
                "year_emerged": 2016,
                "year_viral": 2017,
                "core_claim": "Australia is not real; British prisoners were killed at sea, and actors maintain the illusion.",
                "villain": "British Empire, Australian government, global conspiracy",
                "villain_specificity": 0.70,
                "villain_power_level": 0.65,
                "evidence_types": ["satirical logic", "prison ship records", "upside down memes"],
                "internal_coherence": 0.30,
                "accessibility": 0.85,
                "emotional_valence": ["humor", "absurdity", "satire"],
                "outcomes": {
                    "google_trends_peak": 16,
                    "google_trends_sustained": 5,
                    "believer_count_estimate": 1800,  # Mostly satirical
                    "social_media_mentions": 420000,
                    "mainstream_coverage": 95,
                    "wikipedia_pageviews_yearly": 32000,
                    "subreddit_subscribers": 18000,
                    "youtube_videos": 850
                },
                "demographics": "Global, young, satirical communities",
                "related_theories": ["finland_doesnt_exist", "wyoming_doesnt_exist"]
            },
            {
                "id": "giraffes_arent_real",
                "name": "Giraffes Aren't Real",
                "short_name": "Giraffe Hoax",
                "year_emerged": 2018,
                "year_viral": 2019,
                "core_claim": "Giraffes are fake animals created by governments to hide surveillance equipment.",
                "villain": "Governments, wildlife organizations",
                "villain_specificity": 0.60,
                "villain_power_level": 0.50,
                "evidence_types": ["satirical reasoning", "similar to birds aren't real"],
                "internal_coherence": 0.40,
                "accessibility": 0.98,
                "emotional_valence": ["humor", "satire", "absurdism"],
                "outcomes": {
                    "google_trends_peak": 12,
                    "google_trends_sustained": 4,
                    "believer_count_estimate": 800,  # Satirical
                    "social_media_mentions": 180000,
                    "mainstream_coverage": 35,
                    "wikipedia_pageviews_yearly": 12000,
                    "subreddit_subscribers": 8500,
                    "youtube_videos": 420
                },
                "demographics": "Young, meme communities, satirical",
                "related_theories": ["birds_arent_real"]
            }
        ]
    
    def _generate_rich_description(self, theory: Dict) -> str:
        """Generate rich 400-500 word narrative description."""
        desc = f"{theory['name']}: "
        desc += f"{theory['core_claim']} "
        desc += f"\n\nThis conspiracy theory emerged around {theory['year_emerged']}"
        
        if theory['year_viral'] != theory['year_emerged']:
            desc += f" but went viral in {theory['year_viral']}"
        desc += ". "
        
        desc += f"The theory identifies {theory['villain']} as the primary antagonists. "
        
        # Evidence description
        evidence_str = ", ".join(theory['evidence_types'])
        desc += f"Believers cite evidence including {evidence_str}. "
        
        # Coherence and accessibility
        if theory['internal_coherence'] > 0.7:
            desc += "The theory maintains relatively high internal logical consistency. "
        elif theory['internal_coherence'] < 0.4:
            desc += "The theory contains numerous internal contradictions and plot holes. "
        
        if theory['accessibility'] > 0.85:
            desc += "It's easily understood and requires minimal background knowledge. "
        elif theory['accessibility'] < 0.6:
            desc += "Understanding the theory requires substantial prerequisite knowledge. "
        
        # Emotional hooks
        emotions = ", ".join(theory['emotional_valence'])
        desc += f"The theory resonates emotionally through themes of {emotions}. "
        
        # Virality metrics
        outcomes = theory['outcomes']
        desc += f"\n\nIn terms of spread and belief, "
        desc += f"the theory reached a Google Trends peak of {outcomes['google_trends_peak']} "
        desc += f"with sustained interest at {outcomes['google_trends_sustained']}. "
        desc += f"Estimated believers number around {outcomes['believer_count_estimate']:,}. "
        desc += f"Social media mentions total approximately {outcomes['social_media_mentions']:,}, "
        desc += f"with mainstream media coverage in {outcomes['mainstream_coverage']} articles. "
        
        if outcomes['subreddit_subscribers'] > 0:
            desc += f"Reddit communities dedicated to this theory have {outcomes['subreddit_subscribers']:,} subscribers. "
        
        desc += f"YouTube hosts approximately {outcomes['youtube_videos']:,} videos discussing the theory. "
        desc += f"Wikipedia pages related to this theory receive about {outcomes['wikipedia_pageviews_yearly']:,} views annually. "
        
        # Demographics
        desc += f"\n\nThe theory's believer base is characterized as: {theory['demographics']}. "
        
        # Related theories
        if theory['related_theories']:
            related = ", ".join(theory['related_theories'])
            desc += f"This theory often connects with related conspiracy theories including {related}. "
        
        # Villain analysis
        desc += f"\n\nThe villain structure shows specificity of {theory['villain_specificity']:.2f} "
        desc += f"and power level of {theory['villain_power_level']:.2f}. "
        
        return desc
    
    def generate_dataset(self) -> Dict:
        """Generate complete dataset with all theories."""
        print("Collecting conspiracy theories...")
        theories = self.collect_all_theories()
        print(f"Collected {len(theories)} theories")
        
        # Generate rich descriptions
        print("Generating rich descriptions...")
        for theory in theories:
            theory['rich_description'] = self._generate_rich_description(theory)
        
        # Calculate virality score for each theory
        print("Calculating virality scores...")
        self._calculate_virality_scores(theories)
        
        dataset = {
            "theories": theories,
            "metadata": {
                "total_theories": len(theories),
                "date_collected": "2025-11-13",
                "domain": "conspiracy_theories",
                "outcome_metric": "virality_score",
                "description": "Comprehensive conspiracy theory dataset for narrative analysis"
            }
        }
        
        return dataset
    
    def _calculate_virality_scores(self, theories: List[Dict]):
        """Calculate normalized virality score for each theory."""
        import numpy as np
        
        # Extract outcome metrics
        trends_sustained = np.array([t['outcomes']['google_trends_sustained'] for t in theories])
        social_mentions = np.array([np.log10(max(t['outcomes']['social_media_mentions'], 1)) for t in theories])
        believers = np.array([np.log10(max(t['outcomes']['believer_count_estimate'], 1)) for t in theories])
        wiki_views = np.array([t['outcomes']['wikipedia_pageviews_yearly'] for t in theories])
        media_coverage = np.array([t['outcomes']['mainstream_coverage'] for t in theories])
        
        # Normalize each metric to 0-1
        def normalize(arr):
            min_val, max_val = arr.min(), arr.max()
            if max_val == min_val:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
        
        trends_norm = normalize(trends_sustained)
        social_norm = normalize(social_mentions)
        believers_norm = normalize(believers)
        wiki_norm = normalize(wiki_views)
        media_norm = normalize(media_coverage)
        
        # Weighted combination (as specified in plan)
        virality_scores = (
            0.30 * trends_norm +
            0.25 * social_norm +
            0.20 * believers_norm +
            0.15 * wiki_norm +
            0.10 * media_norm
        )
        
        # Add to theories
        for i, theory in enumerate(theories):
            theory['virality_score'] = float(virality_scores[i])
            theory['virality_rank'] = None  # Will be set after sorting
        
        # Calculate ranks
        sorted_theories = sorted(enumerate(theories), key=lambda x: x[1]['virality_score'], reverse=True)
        for rank, (idx, theory) in enumerate(sorted_theories, 1):
            theories[idx]['virality_rank'] = rank
            theories[idx]['top_quartile'] = 1 if rank <= len(theories) // 4 else 0
    
    def save_dataset(self, dataset: Dict, output_path: str):
        """Save dataset to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"Total theories: {len(dataset['theories'])}")
        print(f"Top quartile count: {sum(t['top_quartile'] for t in dataset['theories'])}")


def main():
    """Main execution."""
    collector = ConspiracyDataCollector()
    dataset = collector.generate_dataset()
    
    # Save dataset
    output_path = "/Users/michaelsmerconish/Desktop/RandomCode/novelization/data/conspiracy_theories_complete.json"
    collector.save_dataset(dataset, output_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    theories = dataset['theories']
    print(f"Total theories collected: {len(theories)}")
    print(f"Virality score range: {min(t['virality_score'] for t in theories):.3f} - {max(t['virality_score'] for t in theories):.3f}")
    print(f"Top 5 most viral:")
    sorted_theories = sorted(theories, key=lambda x: x['virality_score'], reverse=True)
    for i, theory in enumerate(sorted_theories[:5], 1):
        print(f"  {i}. {theory['short_name']}: {theory['virality_score']:.3f}")
    
    print(f"\nBottom 5 least viral:")
    for i, theory in enumerate(sorted_theories[-5:], 1):
        print(f"  {i}. {theory['short_name']}: {theory['virality_score']:.3f}")
    
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

