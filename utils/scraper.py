import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import os
 
# Relevant Wikipedia page titles about Modern History of India
TOPICS = [
    "Human_evolution",
    "Adaptive_evolution_in_the_human_genome",
    "Human_origins",
    "Dual_inheritance_theory",
    "List_of_human_evolution_fossils",
    "Timeline_of_human_evolution",
    "Archaeogenetics",
    "Origin_of_language",
    "Origin_of_speech",
    "Evolutionary_medicine",
    "Evolution_of_morality",
    "Evolutionary_neuroscience",
    "Evolutionary_origin_of_religion",
    "Evolutionary_psychology",
    "Chimpanzee–human_last_common_ancestor",
    "Gorilla–human_last_common_ancestor",
    "Orangutan–human_last_common_ancestor",
    "Gibbon–human_last_common_ancestor",
    "Gathering_hypothesis",
    "Endurance_running_hypothesis",
    "Aquatic_ape_hypothesis",
    "Sexual_selection_in_humans",
    "Bipedalism#Evolution_of_human_bipedalism",
    "Human_skeletal_changes_due_to_bipedalism",
    "Muscular_evolution_in_humans",
    "Hunting_hypothesis",
    "Recent_African_origin_of_modern_humans",
    "Multiregional_origin_of_modern_humans",
    "Interbreeding_between_archaic_and_modern_humans"
    "Behavioral_modernity",
    "Early_human_migrations",
    "Recent_human_evolution",
    "Ardipithecus",
    "Australopithecus",
    "Paranthropus",
    "Homo_erectus",
    "Homo",
    "Human",
    "Cold_and_heat_adaptations_in_humans",
    "Hair#Evolution",
    "Human_evolutionary_developmental_biology",
    "Paleoanthropology",
    "Hominidae",
    "List_of_fictional_primates",
    "Mythic_humanoids",
    "Yeren",
    "List_of_individual_apes",
    "Monkeys_and_apes_in_space"

   
]
 
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
 
SAVE_DIR = "wiki_data"
os.makedirs(SAVE_DIR, exist_ok=True)
 
# Clean the text to remove unnecessary whitespaces
def clean_text(text):
    return ' '.join(text.split())
 
# Scrape the Wikipedia page content
def scrape_wikipedia_page(title):
    url = f"https://en.wikipedia.org/wiki/{title}"
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
 
        content_div = soup.find("div", {"class": "mw-parser-output"})
        paragraphs = content_div.find_all("p")
        
        # Join paragraphs together, excluding very short ones
        text = "\n".join(clean_text(p.get_text()) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        return text
 
    except Exception as e:
        print(f"Error scraping {title}: {e}")
        return ""
 
# Save the data into a text file with unique delimiters
def save_data(topics):
    full_text = ""
    for topic in tqdm(topics):
        text = scrape_wikipedia_page(topic)
        if text:
            full_text += f"\n\n---\n\n# {topic.replace('_', ' ')}\n\n{text}"
        time.sleep(1)  # Be respectful to Wikipedia servers
 
    file_path = os.path.join(SAVE_DIR, "modern_history_of_india.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n✅ Data saved to: {file_path}")
 
if __name__ == "__main__":
    save_data(TOPICS)