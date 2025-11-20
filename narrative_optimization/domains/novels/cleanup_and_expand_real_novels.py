"""
Clean up fake books and expand with real novels only.

Steps:
1. Remove all fake/generated books
2. Keep only real novels with successful enrichment
3. Add comprehensive list of real novels from multiple sources
4. Enrich all new books with real data

Author: Narrative Framework System
Date: November 2025
"""

import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests

try:
    from olclient.openlibrary import OpenLibrary
except ImportError:
    try:
        from openlibrary_client import OpenLibrary
    except:
        OpenLibrary = None

import wikipedia

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_expansion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suspicious patterns for fake books
FAKE_PATTERNS = [
    r'^A\s+(Ancient|Bright|Silent|Hidden|Lost|Forgotten|Distant|Mysterious|Dark)\s+(Hope|Dream|War|Secret|Love|Fear|Lie|Truth|Promise|Journey)',
    r'^The\s+(Ancient|Bright|Silent|Hidden|Lost|Forgotten|Distant|Mysterious|Dark)\s+(Hope|Dream|War|Secret|Love|Fear|Lie|Truth|Promise|Journey)',
    r'^(Emily|Jessica|Sarah|Michael|David|Andrew|Joshua|Amanda|Nicole|Christopher|Stephanie|Daniel|Matthew|Jennifer|Michelle|Ashley|Melissa|Robert|James)\'s\s+(Hope|Dream|War|Secret|Love|Fear|Lie|Truth|Promise|Journey)',
    r'^(Spring|Summer|Autumn|Winter|Dawn|Dusk|Morning|Night|Home|Away)\s+(Hope|Dream|War|Secret|Love|Fear|Lie|Truth|Promise|Journey)',
]

# Comprehensive list of REAL novels to add
REAL_NOVELS_TO_ADD = [
    # Modern Bestsellers (2010s-2020s)
    {'title': 'Normal People', 'author': 'Sally Rooney', 'year': 2018},
    {'title': 'Circe', 'author': 'Madeline Miller', 'year': 2018},
    {'title': 'The Song of Achilles', 'author': 'Madeline Miller', 'year': 2011},
    {'title': 'Eleanor Oliphant Is Completely Fine', 'author': 'Gail Honeyman', 'year': 2017},
    {'title': 'Little Fires Everywhere', 'author': 'Celeste Ng', 'year': 2017},
    {'title': 'The Nightingale', 'author': 'Kristin Hannah', 'year': 2015},
    {'title': 'The Goldfinch', 'author': 'Donna Tartt', 'year': 2013},
    {'title': 'The Overstory', 'author': 'Richard Powers', 'year': 2018},
    {'title': 'Pachinko', 'author': 'Min Jin Lee', 'year': 2017},
    {'title': 'There There', 'author': 'Tommy Orange', 'year': 2018},
    {'title': 'The Nickel Boys', 'author': 'Colson Whitehead', 'year': 2019},
    {'title': 'Such a Fun Age', 'author': 'Kiley Reid', 'year': 2019},
    {'title': 'The Dutch House', 'author': 'Ann Patchett', 'year': 2019},
    {'title': 'An American Marriage', 'author': 'Tayari Jones', 'year': 2018},
    {'title': 'Homegoing', 'author': 'Yaa Gyasi', 'year': 2016},
    {'title': 'The Hate U Give', 'author': 'Angie Thomas', 'year': 2017},
    {'title': 'Educated', 'author': 'Tara Westover', 'year': 2018},
    {'title': 'Daisy Jones & The Six', 'author': 'Taylor Jenkins Reid', 'year': 2019},
    {'title': 'The Vanishing Half', 'author': 'Brit Bennett', 'year': 2020},
    {'title': 'Anxious People', 'author': 'Fredrik Backman', 'year': 2019},
    
    # 2000s Important Novels
    {'title': 'The Corrections', 'author': 'Jonathan Franzen', 'year': 2001},
    {'title': 'Atonement', 'author': 'Ian McEwan', 'year': 2001},
    {'title': 'The Amazing Adventures of Kavalier & Clay', 'author': 'Michael Chabon', 'year': 2000},
    {'title': 'Everything Is Illuminated', 'author': 'Jonathan Safran Foer', 'year': 2002},
    {'title': 'The Namesake', 'author': 'Jhumpa Lahiri', 'year': 2003},
    {'title': 'Cloud Atlas', 'author': 'David Mitchell', 'year': 2004},
    {'title': 'The Book Thief', 'author': 'Markus Zusak', 'year': 2005},
    {'title': 'Half of a Yellow Sun', 'author': 'Chimamanda Ngozi Adichie', 'year': 2006},
    {'title': 'The Yiddish Policemen\'s Union', 'author': 'Michael Chabon', 'year': 2007},
    {'title': '2666', 'author': 'Roberto Bolaño', 'year': 2004},
    
    # Classic 20th Century (filling gaps)
    {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald', 'year': 1925},
    {'title': 'To the Lighthouse', 'author': 'Virginia Woolf', 'year': 1927},
    {'title': 'A Farewell to Arms', 'author': 'Ernest Hemingway', 'year': 1929},
    {'title': 'The Sound and the Fury', 'author': 'William Faulkner', 'year': 1929},
    {'title': 'Their Eyes Were Watching God', 'author': 'Zora Neale Hurston', 'year': 1937},
    {'title': 'Rebecca', 'author': 'Daphne du Maurier', 'year': 1938},
    {'title': 'The Grapes of Wrath', 'author': 'John Steinbeck', 'year': 1939},
    {'title': 'Native Son', 'author': 'Richard Wright', 'year': 1940},
    {'title': 'The Fountainhead', 'author': 'Ayn Rand', 'year': 1943},
    {'title': 'All the King\'s Men', 'author': 'Robert Penn Warren', 'year': 1946},
    
    # 1950s-1970s Classics
    {'title': 'The Adventures of Augie March', 'author': 'Saul Bellow', 'year': 1953},
    {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien', 'year': 1954},
    {'title': 'A Tree Grows in Brooklyn', 'author': 'Betty Smith', 'year': 1943},
    {'title': 'The Old Man and the Sea', 'author': 'Ernest Hemingway', 'year': 1952},
    {'title': 'Fahrenheit 451', 'author': 'Ray Bradbury', 'year': 1953},
    {'title': 'East of Eden', 'author': 'John Steinbeck', 'year': 1952},
    {'title': 'Giovanni\'s Room', 'author': 'James Baldwin', 'year': 1956},
    {'title': 'Atlas Shrugged', 'author': 'Ayn Rand', 'year': 1957},
    {'title': 'Breakfast at Tiffany\'s', 'author': 'Truman Capote', 'year': 1958},
    {'title': 'A Separate Peace', 'author': 'John Knowles', 'year': 1959},
    
    # 1960s-1980s Important Works
    {'title': 'Franny and Zooey', 'author': 'J.D. Salinger', 'year': 1961},
    {'title': 'A Clockwork Orange', 'author': 'Anthony Burgess', 'year': 1962},
    {'title': 'The Golden Notebook', 'author': 'Doris Lessing', 'year': 1962},
    {'title': 'The Group', 'author': 'Mary McCarthy', 'year': 1963},
    {'title': 'A House for Mr Biswas', 'author': 'V.S. Naipaul', 'year': 1961},
    {'title': 'The Spy Who Came in from the Cold', 'author': 'John le Carré', 'year': 1963},
    {'title': 'In Cold Blood', 'author': 'Truman Capote', 'year': 1966},
    {'title': 'The Master and Margarita', 'author': 'Mikhail Bulgakov', 'year': 1967},
    {'title': 'Portnoy\'s Complaint', 'author': 'Philip Roth', 'year': 1969},
    {'title': 'Gravity\'s Rainbow', 'author': 'Thomas Pynchon', 'year': 1973},
    {'title': 'Fear of Flying', 'author': 'Erica Jong', 'year': 1973},
    {'title': 'Ragtime', 'author': 'E.L. Doctorow', 'year': 1975},
    {'title': 'The World According to Garp', 'author': 'John Irving', 'year': 1978},
    {'title': 'Sophie\'s Choice', 'author': 'William Styron', 'year': 1979},
    {'title': 'Midnight\'s Children', 'author': 'Salman Rushdie', 'year': 1981},
    {'title': 'The Color Purple', 'author': 'Alice Walker', 'year': 1982},
    {'title': 'The House of the Spirits', 'author': 'Isabel Allende', 'year': 1982},
    {'title': 'Neuromancer', 'author': 'William Gibson', 'year': 1984},
    {'title': 'Blood Meridian', 'author': 'Cormac McCarthy', 'year': 1985},
    {'title': 'The Bonfire of the Vanities', 'author': 'Tom Wolfe', 'year': 1987},
    
    # 1990s Important Novels
    {'title': 'Possession', 'author': 'A.S. Byatt', 'year': 1990},
    {'title': 'The Things They Carried', 'author': 'Tim O\'Brien', 'year': 1990},
    {'title': 'American Psycho', 'author': 'Bret Easton Ellis', 'year': 1991},
    {'title': 'All the Pretty Horses', 'author': 'Cormac McCarthy', 'year': 1992},
    {'title': 'The Shipping News', 'author': 'E. Annie Proulx', 'year': 1993},
    {'title': 'Snow Falling on Cedars', 'author': 'David Guterson', 'year': 1994},
    {'title': 'The Ghost Road', 'author': 'Pat Barker', 'year': 1995},
    {'title': 'Infinite Jest', 'author': 'David Foster Wallace', 'year': 1996},
    {'title': 'American Gods', 'author': 'Neil Gaiman', 'year': 2001},
    {'title': 'Harry Potter and the Philosopher\'s Stone', 'author': 'J.K. Rowling', 'year': 1997},
    {'title': 'Harry Potter and the Prisoner of Azkaban', 'author': 'J.K. Rowling', 'year': 1999},
    {'title': 'The Blind Assassin', 'author': 'Margaret Atwood', 'year': 2000},
    {'title': 'Girl with a Pearl Earring', 'author': 'Tracy Chevalier', 'year': 1999},
    {'title': 'Memoirs of a Geisha', 'author': 'Arthur Golden', 'year': 1997},
    
    # Science Fiction & Fantasy Classics
    {'title': 'Foundation', 'author': 'Isaac Asimov', 'year': 1951},
    {'title': 'Stranger in a Strange Land', 'author': 'Robert A. Heinlein', 'year': 1961},
    {'title': 'The Left Hand of Darkness', 'author': 'Ursula K. Le Guin', 'year': 1969},
    {'title': 'The Dispossessed', 'author': 'Ursula K. Le Guin', 'year': 1974},
    {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams', 'year': 1979},
    {'title': 'Ender\'s Game', 'author': 'Orson Scott Card', 'year': 1985},
    {'title': 'The Handmaid\'s Tale', 'author': 'Margaret Atwood', 'year': 1985},
    {'title': 'Snow Crash', 'author': 'Neal Stephenson', 'year': 1992},
    {'title': 'The Fifth Season', 'author': 'N.K. Jemisin', 'year': 2015},
    {'title': 'Station Eleven', 'author': 'Emily St. John Mandel', 'year': 2014},
    {'title': 'The Three-Body Problem', 'author': 'Liu Cixin', 'year': 2008},
    
    # Mystery/Thriller Classics
    {'title': 'The Maltese Falcon', 'author': 'Dashiell Hammett', 'year': 1930},
    {'title': 'The Big Sleep', 'author': 'Raymond Chandler', 'year': 1939},
    {'title': 'And Then There Were None', 'author': 'Agatha Christie', 'year': 1939},
    {'title': 'The Day of the Jackal', 'author': 'Frederick Forsyth', 'year': 1971},
    {'title': 'The Girl with the Dragon Tattoo', 'author': 'Stieg Larsson', 'year': 2005},
    {'title': 'Gone Girl', 'author': 'Gillian Flynn', 'year': 2012},
    {'title': 'The Woman in the Window', 'author': 'A.J. Finn', 'year': 2018},
    
    # Historical Fiction
    {'title': 'I, Claudius', 'author': 'Robert Graves', 'year': 1934},
    {'title': 'The Name of the Rose', 'author': 'Umberto Eco', 'year': 1980},
    {'title': 'The Pillars of the Earth', 'author': 'Ken Follett', 'year': 1989},
    {'title': 'The Amazing Adventures of Kavalier & Clay', 'author': 'Michael Chabon', 'year': 2000},
    {'title': 'The Kite Runner', 'author': 'Khaled Hosseini', 'year': 2003},
    {'title': 'A Thousand Splendid Suns', 'author': 'Khaled Hosseini', 'year': 2007},
    {'title': 'The Book Thief', 'author': 'Markus Zusak', 'year': 2005},
    {'title': 'All the Light We Cannot See', 'author': 'Anthony Doerr', 'year': 2014},
    
    # International/World Literature
    {'title': 'The Trial', 'author': 'Franz Kafka', 'year': 1925},
    {'title': 'The Castle', 'author': 'Franz Kafka', 'year': 1926},
    {'title': 'Journey to the End of the Night', 'author': 'Louis-Ferdinand Céline', 'year': 1932},
    {'title': 'The Unbearable Lightness of Being', 'author': 'Milan Kundera', 'year': 1984},
    {'title': 'Like Water for Chocolate', 'author': 'Laura Esquivel', 'year': 1989},
    {'title': 'The God of Small Things', 'author': 'Arundhati Roy', 'year': 1997},
    {'title': 'The Shadow of the Wind', 'author': 'Carlos Ruiz Zafón', 'year': 2001},
    {'title': 'The Kite Runner', 'author': 'Khaled Hosseini', 'year': 2003},
    {'title': 'Never Let Me Go', 'author': 'Kazuo Ishiguro', 'year': 2005},
    {'title': 'The Brief Wondrous Life of Oscar Wao', 'author': 'Junot Díaz', 'year': 2007},
    {'title': 'A Man Called Ove', 'author': 'Fredrik Backman', 'year': 2012},
    {'title': 'My Brilliant Friend', 'author': 'Elena Ferrante', 'year': 2011},
    
    # 19th Century Classics (filling gaps)
    {'title': 'Frankenstein', 'author': 'Mary Shelley', 'year': 1818},
    {'title': 'The Scarlet Letter', 'author': 'Nathaniel Hawthorne', 'year': 1850},
    {'title': 'Madame Bovary', 'author': 'Gustave Flaubert', 'year': 1856},
    {'title': 'The Woman in White', 'author': 'Wilkie Collins', 'year': 1859},
    {'title': 'Les Misérables', 'author': 'Victor Hugo', 'year': 1862},
    {'title': 'Crime and Punishment', 'author': 'Fyodor Dostoevsky', 'year': 1866},
    {'title': 'The Brothers Karamazov', 'author': 'Fyodor Dostoevsky', 'year': 1880},
    {'title': 'The Portrait of a Lady', 'author': 'Henry James', 'year': 1881},
    {'title': 'The Strange Case of Dr Jekyll and Mr Hyde', 'author': 'Robert Louis Stevenson', 'year': 1886},
    {'title': 'Tess of the d\'Urbervilles', 'author': 'Thomas Hardy', 'year': 1891},
    {'title': 'The Awakening', 'author': 'Kate Chopin', 'year': 1899},
    {'title': 'Sister Carrie', 'author': 'Theodore Dreiser', 'year': 1900},
    
    # Contemporary (2020s)
    {'title': 'The Midnight Library', 'author': 'Matt Haig', 'year': 2020},
    {'title': 'Hamnet', 'author': 'Maggie O\'Farrell', 'year': 2020},
    {'title': 'The Invisible Life of Addie LaRue', 'author': 'V.E. Schwab', 'year': 2020},
    {'title': 'Klara and the Sun', 'author': 'Kazuo Ishiguro', 'year': 2021},
    {'title': 'The Lincoln Highway', 'author': 'Amor Towles', 'year': 2021},
    {'title': 'Cloud Cuckoo Land', 'author': 'Anthony Doerr', 'year': 2021},
    {'title': 'The Seven Husbands of Evelyn Hugo', 'author': 'Taylor Jenkins Reid', 'year': 2017},
    {'title': 'Tomorrow, and Tomorrow, and Tomorrow', 'author': 'Gabrielle Zevin', 'year': 2022},
    {'title': 'Demon Copperhead', 'author': 'Barbara Kingsolver', 'year': 2022},
    {'title': 'Lessons in Chemistry', 'author': 'Bonnie Garmus', 'year': 2022},
]


class NovelsCleanupAndExpansion:
    """Clean fake books and expand with real novels."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.novels = []
        self.ol = None
        
        try:
            self.ol = OpenLibrary()
            logger.info("Open Library client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Open Library: {e}")
    
    def is_fake_book(self, novel: Dict[str, Any]) -> bool:
        """Determine if a book is fake based on patterns and enrichment status."""
        title = novel.get('title', '')
        author = novel.get('author', '')
        
        # Check if enrichment failed
        if not novel.get('data_enriched', False):
            return True
        
        # Check for generic template in plot summary
        plot = novel.get('plot_summary', '')
        if 'celebrated work of fiction' in plot.lower() and 'explores themes' in plot.lower():
            return True
        
        # Check against fake patterns
        for pattern in FAKE_PATTERNS:
            if re.match(pattern, title, re.IGNORECASE):
                logger.debug(f"Fake pattern matched: {title}")
                return True
        
        # Check for generic author names with common first/last names
        common_first = ['Emily', 'Jessica', 'Sarah', 'Michael', 'David', 'Andrew', 'Joshua', 
                       'Amanda', 'Nicole', 'Christopher', 'Stephanie', 'Daniel', 'Matthew',
                       'Jennifer', 'Michelle', 'Ashley', 'Melissa', 'Robert', 'James', 'Ryan']
        common_last = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                      'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Wilson',
                      'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee']
        
        author_parts = author.split()
        if (len(author_parts) == 2 and 
            author_parts[0] in common_first and 
            author_parts[1] in common_last and
            len(plot) < 500):  # Short plot suggests poor data
            return True
        
        return False
    
    def cleanup_dataset(self):
        """Remove fake books, keep only real ones."""
        logger.info("="*80)
        logger.info("CLEANING UP FAKE BOOKS")
        logger.info("="*80)
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.novels = json.load(f)
        
        logger.info(f"Starting with {len(self.novels)} books")
        
        real_novels = []
        fake_count = 0
        
        for novel in self.novels:
            if self.is_fake_book(novel):
                fake_count += 1
                logger.debug(f"Removing fake: {novel.get('title')} by {novel.get('author')}")
            else:
                real_novels.append(novel)
                logger.debug(f"Keeping real: {novel.get('title')} by {novel.get('author')}")
        
        self.novels = real_novels
        
        logger.info(f"Removed {fake_count} fake books")
        logger.info(f"Kept {len(self.novels)} real books")
        
        return self.novels
    
    def add_new_real_novels(self):
        """Add new real novels from comprehensive list."""
        logger.info("="*80)
        logger.info("ADDING NEW REAL NOVELS")
        logger.info("="*80)
        
        # Get existing titles to avoid duplicates
        existing_titles = set()
        for novel in self.novels:
            title_key = f"{novel.get('title', '').lower().strip()}_{novel.get('author', '').lower().strip()}"
            existing_titles.add(title_key)
        
        new_novels = []
        for book_info in REAL_NOVELS_TO_ADD:
            title_key = f"{book_info['title'].lower().strip()}_{book_info['author'].lower().strip()}"
            
            if title_key not in existing_titles:
                novel = {
                    'title': book_info['title'],
                    'author': book_info['author'],
                    'publication_year': book_info['year'],
                    'source': 'canonical_literature',
                    'data_enriched': False
                }
                new_novels.append(novel)
                existing_titles.add(title_key)
                logger.info(f"Adding: {book_info['title']} by {book_info['author']}")
            else:
                logger.debug(f"Already exists: {book_info['title']}")
        
        logger.info(f"Added {len(new_novels)} new novels to enrich")
        self.novels.extend(new_novels)
        
        return len(new_novels)
    
    def enrich_novel(self, novel: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Enrich a single novel with real data (same as before)."""
        title = novel.get('title', '')
        author = novel.get('author', '')
        
        logger.info(f"Processing: {title} by {author}")
        
        # Fetch from Wikipedia
        wiki_data = self.fetch_from_wikipedia(title, author)
        time.sleep(0.5)
        
        # Fetch from Google Books
        gb_data = self.fetch_from_google_books(title, author)
        time.sleep(0.5)
        
        has_data = bool(wiki_data or gb_data)
        
        if has_data:
            # Choose best description
            best_description = None
            if wiki_data and 'description' in wiki_data:
                best_description = wiki_data['description']
            elif gb_data and 'description' in gb_data:
                best_description = gb_data['description']
            
            if best_description:
                novel['plot_summary'] = best_description
                novel['full_narrative'] = best_description
                
                # Extract character names
                character_names = self.extract_character_names(best_description)
                novel['character_names'] = character_names
            
            # Extract genres
            genres = []
            if gb_data and 'categories' in gb_data:
                genres.extend(gb_data['categories'])
            novel['genres'] = list(set(genres))[:10]
            
            # Add metadata
            if gb_data:
                if 'publisher' in gb_data:
                    novel['publisher'] = gb_data['publisher']
                if 'page_count' in gb_data:
                    novel['page_count'] = gb_data['page_count']
            
            if wiki_data and 'wikipedia_url' in wiki_data:
                novel['wikipedia_url'] = wiki_data['wikipedia_url']
            
            novel['data_enriched'] = True
            novel['enrichment_sources'] = []
            if wiki_data:
                novel['enrichment_sources'].append('wikipedia')
            if gb_data:
                novel['enrichment_sources'].append('google_books')
            
            logger.info(f"✓ Successfully enriched: {title}")
            return novel, True
        else:
            logger.warning(f"✗ No data found for: {title}")
            novel['data_enriched'] = False
            return novel, False
    
    def fetch_from_wikipedia(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Fetch book data from Wikipedia."""
        try:
            search_queries = [
                f"{title} (novel)",
                f"{title} ({author})",
                title
            ]
            
            for query in search_queries:
                try:
                    page = wikipedia.page(query, auto_suggest=False)
                    summary = page.summary
                    
                    if len(summary) > 200:
                        return {
                            'description': summary,
                            'wikipedia_url': page.url
                        }
                except wikipedia.exceptions.DisambiguationError as e:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        summary = page.summary
                        if len(summary) > 200 and (title.lower() in summary.lower() or author.lower() in summary.lower()):
                            return {
                                'description': summary,
                                'wikipedia_url': page.url
                            }
                    except:
                        continue
                except wikipedia.exceptions.PageError:
                    continue
            return None
        except Exception as e:
            logger.debug(f"Wikipedia error for {title}: {e}")
            return None
    
    def fetch_from_google_books(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Fetch book data from Google Books API."""
        try:
            query = f"intitle:{title} inauthor:{author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if 'items' not in data or not data['items']:
                return None
            
            volume = data['items'][0]['volumeInfo']
            result = {}
            
            if 'description' in volume:
                result['description'] = volume['description']
            if 'categories' in volume:
                result['categories'] = volume['categories']
            if 'publisher' in volume:
                result['publisher'] = volume['publisher']
            if 'pageCount' in volume:
                result['page_count'] = volume['pageCount']
            
            return result if result else None
        except Exception as e:
            logger.debug(f"Google Books error for {title}: {e}")
            return None
    
    def extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text."""
        if not text:
            return []
        
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        common_words = {'The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For',
                       'New', 'York', 'Times', 'Book', 'Novel', 'Story', 'Chapter'}
        
        character_names = []
        seen = set()
        
        for match in matches:
            name = match.strip()
            if len(name) >= 3 and name not in common_words and name.lower() not in seen:
                character_names.append(name)
                seen.add(name.lower())
                if len(character_names) >= 20:
                    break
        
        return character_names
    
    def enrich_new_novels(self):
        """Enrich all novels that haven't been enriched yet."""
        logger.info("="*80)
        logger.info("ENRICHING NEW NOVELS")
        logger.info("="*80)
        
        to_enrich = [n for n in self.novels if not n.get('data_enriched', False)]
        logger.info(f"Found {len(to_enrich)} novels to enrich")
        
        success_count = 0
        for i, novel in enumerate(to_enrich):
            _, success = self.enrich_novel(novel)
            if success:
                success_count += 1
            
            if (i + 1) % 50 == 0:
                self.save_dataset()
                logger.info(f"Progress: {i + 1}/{len(to_enrich)} ({success_count} successful)")
        
        logger.info(f"Enrichment complete: {success_count}/{len(to_enrich)} successful")
    
    def save_dataset(self):
        """Save dataset to file."""
        logger.info(f"Saving dataset with {len(self.novels)} novels")
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.novels, f, indent=2, ensure_ascii=False)
        logger.info("Dataset saved successfully")
    
    def run_full_cleanup_and_expansion(self):
        """Run complete cleanup and expansion process."""
        logger.info("="*80)
        logger.info("STARTING FULL CLEANUP AND EXPANSION")
        logger.info("="*80)
        
        # Step 1: Cleanup fake books
        self.cleanup_dataset()
        self.save_dataset()
        
        # Step 2: Add new real novels
        new_count = self.add_new_real_novels()
        self.save_dataset()
        
        # Step 3: Enrich all new novels
        if new_count > 0:
            self.enrich_new_novels()
            self.save_dataset()
        
        # Final summary
        enriched_count = sum(1 for n in self.novels if n.get('data_enriched', False))
        logger.info("="*80)
        logger.info("CLEANUP AND EXPANSION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total novels in dataset: {len(self.novels)}")
        logger.info(f"Successfully enriched: {enriched_count}")
        logger.info(f"Success rate: {100 * enriched_count / len(self.novels):.1f}%")


def main():
    dataset_path = Path(__file__).parent / 'data' / 'novels_dataset.json'
    
    cleaner = NovelsCleanupAndExpansion(str(dataset_path))
    cleaner.run_full_cleanup_and_expansion()
    
    logger.info("All done!")


if __name__ == '__main__':
    main()

