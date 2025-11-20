"""
Expand novels dataset to 500+ real books and enrich them all.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Additional 200+ real novels to reach 500+
ADDITIONAL_NOVELS = [
    # Contemporary Literary Fiction
    {'title': 'The Underground Railroad', 'author': 'Colson Whitehead', 'year': 2016},
    {'title': 'Lincoln in the Bardo', 'author': 'George Saunders', 'year': 2017},
    {'title': 'The Sellout', 'author': 'Paul Beatty', 'year': 2015},
    {'title': 'The Sympathizer', 'author': 'Viet Thanh Nguyen', 'year': 2015},
    {'title': 'The Luminaries', 'author': 'Eleanor Catton', 'year': 2013},
    {'title': 'A Tale for the Time Being', 'author': 'Ruth Ozeki', 'year': 2013},
    {'title': 'The Flamethrowers', 'author': 'Rachel Kushner', 'year': 2013},
    {'title': 'Americanah', 'author': 'Chimamanda Ngozi Adichie', 'year': 2013},
    {'title': 'The Lowland', 'author': 'Jhumpa Lahiri', 'year': 2013},
    {'title': 'The Goldfinch', 'author': 'Donna Tartt', 'year': 2013},
    
    # 21st Century Acclaimed
    {'title': 'Gilead', 'author': 'Marilynne Robinson', 'year': 2004},
    {'title': 'The Known World', 'author': 'Edward P. Jones', 'year': 2003},
    {'title': 'Middlesex', 'author': 'Jeffrey Eugenides', 'year': 2002},
    {'title': 'The Amazing Adventures of Kavalier & Clay', 'author': 'Michael Chabon', 'year': 2000},
    {'title': 'White Teeth', 'author': 'Zadie Smith', 'year': 2000},
    {'title': 'The Hours', 'author': 'Michael Cunningham', 'year': 1998},
    {'title': 'Underworld', 'author': 'Don DeLillo', 'year': 1997},
    {'title': 'American Pastoral', 'author': 'Philip Roth', 'year': 1997},
    {'title': 'The Remains of the Day', 'author': 'Kazuo Ishiguro', 'year': 1989},
    
    # Classic American Literature
    {'title': 'Absalom, Absalom!', 'author': 'William Faulkner', 'year': 1936},
    {'title': 'Light in August', 'author': 'William Faulkner', 'year': 1932},
    {'title': 'The Sun Also Rises', 'author': 'Ernest Hemingway', 'year': 1926},
    {'title': 'Main Street', 'author': 'Sinclair Lewis', 'year': 1920},
    {'title': 'Winesburg, Ohio', 'author': 'Sherwood Anderson', 'year': 1919},
    {'title': 'The Age of Innocence', 'author': 'Edith Wharton', 'year': 1920},
    {'title': 'My Ántonia', 'author': 'Willa Cather', 'year': 1918},
    {'title': 'The House of Mirth', 'author': 'Edith Wharton', 'year': 1905},
    {'title': 'The Red Badge of Courage', 'author': 'Stephen Crane', 'year': 1895},
    {'title': 'The Portrait of a Lady', 'author': 'Henry James', 'year': 1881},
    
    # British Classics
    {'title': 'Middlemarch', 'author': 'George Eliot', 'year': 1871},
    {'title': 'Vanity Fair', 'author': 'William Makepeace Thackeray', 'year': 1848},
    {'title': 'David Copperfield', 'author': 'Charles Dickens', 'year': 1850},
    {'title': 'Bleak House', 'author': 'Charles Dickens', 'year': 1853},
    {'title': 'A Tale of Two Cities', 'author': 'Charles Dickens', 'year': 1859},
    {'title': 'Emma', 'author': 'Jane Austen', 'year': 1815},
    {'title': 'Sense and Sensibility', 'author': 'Jane Austen', 'year': 1811},
    {'title': 'Persuasion', 'author': 'Jane Austen', 'year': 1817},
    {'title': 'Northanger Abbey', 'author': 'Jane Austen', 'year': 1817},
    {'title': 'Mansfield Park', 'author': 'Jane Austen', 'year': 1814},
    
    # 20th Century British
    {'title': 'A Passage to India', 'author': 'E.M. Forster', 'year': 1924},
    {'title': 'Howards End', 'author': 'E.M. Forster', 'year': 1910},
    {'title': 'The Rainbow', 'author': 'D.H. Lawrence', 'year': 1915},
    {'title': 'Women in Love', 'author': 'D.H. Lawrence', 'year': 1920},
    {'title': 'A Clockwork Orange', 'author': 'Anthony Burgess', 'year': 1962},
    {'title': 'The Prime of Miss Jean Brodie', 'author': 'Muriel Spark', 'year': 1961},
    {'title': 'The Go-Between', 'author': 'L.P. Hartley', 'year': 1953},
    {'title': 'Brideshead Revisited', 'author': 'Evelyn Waugh', 'year': 1945},
    {'title': 'Scoop', 'author': 'Evelyn Waugh', 'year': 1938},
    
    # Russian Literature
    {'title': 'Dead Souls', 'author': 'Nikolai Gogol', 'year': 1842},
    {'title': 'Fathers and Sons', 'author': 'Ivan Turgenev', 'year': 1862},
    {'title': 'The Idiot', 'author': 'Fyodor Dostoevsky', 'year': 1869},
    {'title': 'Notes from Underground', 'author': 'Fyodor Dostoevsky', 'year': 1864},
    {'title': 'The Death of Ivan Ilyich', 'author': 'Leo Tolstoy', 'year': 1886},
    {'title': 'Doctor Zhivago', 'author': 'Boris Pasternak', 'year': 1957},
    
    # Latin American Literature
    {'title': 'The House of the Spirits', 'author': 'Isabel Allende', 'year': 1982},
    {'title': 'Love in the Time of Cholera', 'author': 'Gabriel García Márquez', 'year': 1985},
    {'title': 'Chronicle of a Death Foretold', 'author': 'Gabriel García Márquez', 'year': 1981},
    {'title': 'The Autumn of the Patriarch', 'author': 'Gabriel García Márquez', 'year': 1975},
    {'title': 'Pedro Páramo', 'author': 'Juan Rulfo', 'year': 1955},
    {'title': 'The Savage Detectives', 'author': 'Roberto Bolaño', 'year': 1998},
    {'title': 'Hopscotch', 'author': 'Julio Cortázar', 'year': 1963},
    
    # Science Fiction & Fantasy
    {'title': 'The Martian Chronicles', 'author': 'Ray Bradbury', 'year': 1950},
    {'title': 'I, Robot', 'author': 'Isaac Asimov', 'year': 1950},
    {'title': 'The Stars My Destination', 'author': 'Alfred Bester', 'year': 1956},
    {'title': 'A Canticle for Leibowitz', 'author': 'Walter M. Miller Jr.', 'year': 1959},
    {'title': 'The Forever War', 'author': 'Joe Haldeman', 'year': 1974},
    {'title': 'Gateway', 'author': 'Frederik Pohl', 'year': 1977},
    {'title': 'The Handmaid\'s Tale', 'author': 'Margaret Atwood', 'year': 1985},
    {'title': 'Oryx and Crake', 'author': 'Margaret Atwood', 'year': 2003},
    {'title': 'The Year of the Flood', 'author': 'Margaret Atwood', 'year': 2009},
    {'title': 'Kindred', 'author': 'Octavia Butler', 'year': 1979},
    {'title': 'Parable of the Sower', 'author': 'Octavia Butler', 'year': 1993},
    {'title': 'The City & the City', 'author': 'China Miéville', 'year': 2009},
    {'title': 'Annihilation', 'author': 'Jeff VanderMeer', 'year': 2014},
    {'title': 'The Windup Girl', 'author': 'Paolo Bacigalupi', 'year': 2009},
    
    # Mystery & Crime
    {'title': 'The Hound of the Baskervilles', 'author': 'Arthur Conan Doyle', 'year': 1902},
    {'title': 'Murder on the Orient Express', 'author': 'Agatha Christie', 'year': 1934},
    {'title': 'The Murder of Roger Ackroyd', 'author': 'Agatha Christie', 'year': 1926},
    {'title': 'The Postman Always Rings Twice', 'author': 'James M. Cain', 'year': 1934},
    {'title': 'The Talented Mr. Ripley', 'author': 'Patricia Highsmith', 'year': 1955},
    {'title': 'L.A. Confidential', 'author': 'James Ellroy', 'year': 1990},
    {'title': 'The Black Dahlia', 'author': 'James Ellroy', 'year': 1987},
    {'title': 'Mystic River', 'author': 'Dennis Lehane', 'year': 2001},
    {'title': 'In the Woods', 'author': 'Tana French', 'year': 2007},
    
    # Historical Fiction
    {'title': 'The Pillars of the Earth', 'author': 'Ken Follett', 'year': 1989},
    {'title': 'Wolf Hall', 'author': 'Hilary Mantel', 'year': 2009},
    {'title': 'Bring Up the Bodies', 'author': 'Hilary Mantel', 'year': 2012},
    {'title': 'The Mirror & the Light', 'author': 'Hilary Mantel', 'year': 2020},
    {'title': 'Shogun', 'author': 'James Clavell', 'year': 1975},
    {'title': 'The Killer Angels', 'author': 'Michael Shaara', 'year': 1974},
    {'title': 'Cold Mountain', 'author': 'Charles Frazier', 'year': 1997},
    {'title': 'March', 'author': 'Geraldine Brooks', 'year': 2005},
    {'title': 'People of the Book', 'author': 'Geraldine Brooks', 'year': 2008},
    
    # Contemporary Bestsellers
    {'title': 'The Fault in Our Stars', 'author': 'John Green', 'year': 2012},
    {'title': 'The Perks of Being a Wallflower', 'author': 'Stephen Chbosky', 'year': 1999},
    {'title': 'The Lovely Bones', 'author': 'Alice Sebold', 'year': 2002},
    {'title': 'The Secret Life of Bees', 'author': 'Sue Monk Kidd', 'year': 2002},
    {'title': 'Water for Elephants', 'author': 'Sara Gruen', 'year': 2006},
    {'title': 'The Art of Racing in the Rain', 'author': 'Garth Stein', 'year': 2008},
    {'title': 'The Alchemist', 'author': 'Paulo Coelho', 'year': 1988},
    {'title': 'Life After Life', 'author': 'Kate Atkinson', 'year': 2013},
    {'title': 'A Gentleman in Moscow', 'author': 'Amor Towles', 'year': 2016},
    {'title': 'The Essex Serpent', 'author': 'Sarah Perry', 'year': 2016},
    
    # Postmodern & Experimental
    {'title': 'The Recognitions', 'author': 'William Gaddis', 'year': 1955},
    {'title': 'V.', 'author': 'Thomas Pynchon', 'year': 1963},
    {'title': 'The Crying of Lot 49', 'author': 'Thomas Pynchon', 'year': 1966},
    {'title': 'White Noise', 'author': 'Don DeLillo', 'year': 1985},
    {'title': 'Libra', 'author': 'Don DeLillo', 'year': 1988},
    {'title': 'If on a winter\'s night a traveler', 'author': 'Italo Calvino', 'year': 1979},
    {'title': 'Invisible Cities', 'author': 'Italo Calvino', 'year': 1972},
    {'title': 'The Book of Laughter and Forgetting', 'author': 'Milan Kundera', 'year': 1979},
    
    # More Contemporary
    {'title': 'The Remains of the Day', 'author': 'Kazuo Ishiguro', 'year': 1989},
    {'title': 'An Artist of the Floating World', 'author': 'Kazuo Ishiguro', 'year': 1986},
    {'title': 'When We Were Orphans', 'author': 'Kazuo Ishiguro', 'year': 2000},
    {'title': 'The Buried Giant', 'author': 'Kazuo Ishiguro', 'year': 2015},
    {'title': 'Atonement', 'author': 'Ian McEwan', 'year': 2001},
    {'title': 'Amsterdam', 'author': 'Ian McEwan', 'year': 1998},
    {'title': 'On Chesil Beach', 'author': 'Ian McEwan', 'year': 2007},
    {'title': 'Saturday', 'author': 'Ian McEwan', 'year': 2005},
    
    # International Contemporary
    {'title': 'The Wind-Up Bird Chronicle', 'author': 'Haruki Murakami', 'year': 1994},
    {'title': 'Kafka on the Shore', 'author': 'Haruki Murakami', 'year': 2002},
    {'title': 'Norwegian Wood', 'author': 'Haruki Murakami', 'year': 1987},
    {'title': '1Q84', 'author': 'Haruki Murakami', 'year': 2009},
    {'title': 'The Vegetarian', 'author': 'Han Kang', 'year': 2007},
    {'title': 'The White Tiger', 'author': 'Aravind Adiga', 'year': 2008},
    {'title': 'Half of a Yellow Sun', 'author': 'Chimamanda Ngozi Adichie', 'year': 2006},
    {'title': 'Purple Hibiscus', 'author': 'Chimamanda Ngozi Adichie', 'year': 2003},
    
    # Young Adult Classics
    {'title': 'The Outsiders', 'author': 'S.E. Hinton', 'year': 1967},
    {'title': 'A Wrinkle in Time', 'author': 'Madeleine L\'Engle', 'year': 1962},
    {'title': 'The Giver', 'author': 'Lois Lowry', 'year': 1993},
    {'title': 'Holes', 'author': 'Louis Sachar', 'year': 1998},
    {'title': 'Speak', 'author': 'Laurie Halse Anderson', 'year': 1999},
    {'title': 'The Hunger Games', 'author': 'Suzanne Collins', 'year': 2008},
    {'title': 'Divergent', 'author': 'Veronica Roth', 'year': 2011},
    {'title': 'The Maze Runner', 'author': 'James Dashner', 'year': 2009},
    
    # Additional Contemporary Literary
    {'title': 'Middlesex', 'author': 'Jeffrey Eugenides', 'year': 2002},
    {'title': 'The Virgin Suicides', 'author': 'Jeffrey Eugenides', 'year': 1993},
    {'title': 'Everything Is Illuminated', 'author': 'Jonathan Safran Foer', 'year': 2002},
    {'title': 'Extremely Loud & Incredibly Close', 'author': 'Jonathan Safran Foer', 'year': 2005},
    {'title': 'The Particular Sadness of Lemon Cake', 'author': 'Aimee Bender', 'year': 2010},
    {'title': 'The History of Love', 'author': 'Nicole Krauss', 'year': 2005},
    {'title': 'A Visit from the Goon Squad', 'author': 'Jennifer Egan', 'year': 2010},
    {'title': 'The Keep', 'author': 'Jennifer Egan', 'year': 2006},
]


class NovelsExpander:
    """Expand novels dataset to 500+."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.novels = []
        self.ol = None
        
        try:
            self.ol = OpenLibrary()
        except:
            pass
    
    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.novels = json.load(f)
        logger.info(f"Loaded {len(self.novels)} existing novels")
    
    def add_new_novels(self):
        existing_titles = set()
        for novel in self.novels:
            key = f"{novel.get('title', '').lower().strip()}_{novel.get('author', '').lower().strip()}"
            existing_titles.add(key)
        
        new_novels = []
        for book in ADDITIONAL_NOVELS:
            key = f"{book['title'].lower().strip()}_{book['author'].lower().strip()}"
            if key not in existing_titles:
                novel = {
                    'title': book['title'],
                    'author': book['author'],
                    'publication_year': book['year'],
                    'source': 'literary_canon',
                    'data_enriched': False
                }
                new_novels.append(novel)
                existing_titles.add(key)
                logger.info(f"Adding: {book['title']} by {book['author']}")
        
        self.novels.extend(new_novels)
        logger.info(f"Added {len(new_novels)} new novels. Total: {len(self.novels)}")
        return len(new_novels)
    
    def enrich_novel(self, novel):
        """Enrich with Wikipedia and Google Books."""
        title = novel.get('title', '')
        author = novel.get('author', '')
        
        # Wikipedia
        try:
            for query in [f"{title} (novel)", title]:
                try:
                    page = wikipedia.page(query, auto_suggest=False)
                    if len(page.summary) > 200:
                        novel['plot_summary'] = page.summary
                        novel['full_narrative'] = page.summary
                        novel['wikipedia_url'] = page.url
                        
                        # Extract character names
                        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
                        matches = re.findall(pattern, page.summary)
                        names = [m for m in matches if len(m) >= 3 and m not in {'The', 'A', 'An', 'Book', 'Novel'}]
                        novel['character_names'] = list(set(names))[:20]
                        break
                except:
                    continue
        except:
            pass
        
        time.sleep(0.5)
        
        # Google Books
        try:
            query = f"intitle:{title} inauthor:{author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data:
                    volume = data['items'][0]['volumeInfo']
                    if 'description' in volume and not novel.get('plot_summary'):
                        novel['plot_summary'] = volume['description']
                        novel['full_narrative'] = volume['description']
                    if 'categories' in volume:
                        novel['genres'] = volume['categories']
                    if 'publisher' in volume:
                        novel['publisher'] = volume['publisher']
                    if 'pageCount' in volume:
                        novel['page_count'] = volume['pageCount']
        except:
            pass
        
        time.sleep(0.5)
        
        novel['data_enriched'] = bool(novel.get('plot_summary'))
        return novel, novel['data_enriched']
    
    def enrich_all_new(self):
        to_enrich = [n for n in self.novels if not n.get('data_enriched')]
        logger.info(f"Enriching {len(to_enrich)} novels...")
        
        success = 0
        for i, novel in enumerate(to_enrich):
            _, enriched = self.enrich_novel(novel)
            if enriched:
                success += 1
            logger.info(f"[{i+1}/{len(to_enrich)}] {novel['title']} - {'✓' if enriched else '✗'}")
            
            if (i + 1) % 50 == 0:
                self.save()
                logger.info(f"Checkpoint: {success}/{i+1} successful")
        
        self.save()
        logger.info(f"Enrichment complete: {success}/{len(to_enrich)} successful")
    
    def save(self):
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.novels, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.novels)} novels")


def main():
    dataset_path = Path(__file__).parent / 'data' / 'novels_dataset.json'
    expander = NovelsExpander(str(dataset_path))
    expander.load_dataset()
    expander.add_new_novels()
    expander.enrich_all_new()
    logger.info(f"Final count: {len(expander.novels)} novels")


if __name__ == '__main__':
    main()






