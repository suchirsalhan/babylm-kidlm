import os
import re
import unicodedata

""" Script used to clean the data. """
!pip3 install nltk
import os
import re
from nltk import tokenize

import re
import unicodedata

def normalize_typography(text):
    # Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)

    # Normalize quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-")

    # Normalize ellipsis
    text = re.sub(r'\.{2,}', '...', text)

    # Normalize repeated punctuation (e.g., !!! → !)
    text = re.sub(r'([?.!]){2,}', r'\1', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s([?.!,;:\'\"\)\]\}])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([(\[\{])\s', r'\1', text)             # Remove space after opening brackets

    return text.strip()

def is_extraneous_line(line):
    line = line.strip()

    # Remove page numbers (e.g., lines with only digits or "Page 12")
    if re.match(r'^(page\s*)?\d+$', line.lower()):
        return True

    # Remove bibliography entries (often start with numbers or authors' names - rough heuristic)
    # Here we just drop lines that look like citation refs or start with [digits]
    if re.match(r'^\[\d+\]', line):
        return True

    # Remove plain text tables - heuristic: lines with many tabs/spaces and digits
    if len(re.findall(r'[\t ]', line)) > 4 and len(re.findall(r'\d', line)) > 2:
        return True

    # Remove one-word on-screen actions in parentheses or brackets, e.g., (laughs), [sigh]
    if re.match(r'^[\(\[]\s*\w+\s*[\)\]]$', line.lower()):
        return True

    # Remove very short lines (<=1 word) likely irrelevant
    if len(line.split()) <= 1:
        return True

    return False

def clean_and_concatenate_lines(lines, concat_size=5, lowercase=True, skip_concat_for_bnc=False, is_bnc=False):
    cleaned_lines = []

    # Step 1: filter extraneous lines
    filtered = []
    for line in lines:
        if not is_extraneous_line(line):
            filtered.append(line.strip())

    # Step 2: normalize and lowercase
    normed = []
    for line in filtered:
        line = normalize_typography(line)
        if lowercase:
            line = line.lower()
        normed.append(line)

    # Step 3: concatenate every concat_size lines unless is_bnc and skip_concat_for_bnc=True
    if is_bnc and skip_concat_for_bnc:
        # Return as-is
        return normed

    concatenated = []
    for i in range(0, len(normed), concat_size):
        group = normed[i:i+concat_size]
        concatenated.append(' '.join(group))

    return concatenated

def clean_aochildes(lines):
    """ For aochildes, we remove the space between the punctuation mark and the final word and join together every 5 lines """
    new_lines = []
    joined = []
    for i, line in enumerate(lines):
        new_line = line[:-3] + line[-2:]
        joined.append(new_line.strip())
        if i % 5 == 0:
            new_lines.append(" ".join(joined) + "\n")
            joined = []
    return new_lines

def clean_bnc_spoken(lines):
    """ For bnc_spoken, we lowercase """
    new_lines = []
    for line in lines:
        new_line = line.lower()
        if new_line != '\n':
            new_lines.append(new_line)
    return new_lines

def clean_cbt(lines):
    """ For cbt, we lowercase and normalise punctuation """
    punctuation = ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '“', '”', '—', '–']
    new_lines = []
    for line in lines:
        new_line = line.lower()
        new_line = new_line.replace(": ' ", ":  \"")
        new_line = new_line.replace("''", "\"")
        new_line = new_line.replace(" '\n", "\"\n")
        new_line = new_line.replace(" ' ", "\" ")
        new_line = new_line.replace(" `` ", "  \"")
        new_line = new_line.replace("` ", " \"")
        new_line = new_line.replace("`", "\"")
        new_line = new_line.replace("’", "\"")
        for punct in punctuation:
            new_line = new_line.replace(f" {punct}", punct)
        new_lines.append(new_line)
    return new_lines

def clean_children_stories(lines):
    """ For children_stories, we lowercase """
    new_lines = []
    for line in lines:
        new_line = line.lower().strip()
        if new_line != '':
            new_lines.append(new_line + "\n")
    return new_lines

def clean_gutenberg(lines):
    """ For gutenberg, we lowercase, remove italics and group lines into paragraphs. We also remove any lines containing '*' or 'p.' """
    # Get paragraphs
    paragraphs = []
    paragraph = ""
    for line in lines:
        # Remove italics
        tmp_line = line.lower().strip().replace('_','')
        if tmp_line == "" and paragraph != "":
            if len(paragraph.split()) > 2 and not paragraph.split()[-1][-1].isnumeric(): # Remove paragraphs with less than 3 words and those that end in a number (probably part of a bibliography)
                paragraphs.append(paragraph[:-1] + '\n')
            paragraph = ""
        else:
            paragraph += tmp_line + " "
    
    # Bad characters - gutenberg has a lot of figures, footnotes, chapter names etc that we want to remove
    bad_chars = ['*', 'p.', '=', '|', '[', ']', '       ', '    ', 'v.']
    new_lines = [p.strip()+'\n' for p in paragraphs if not any([c in p for c in bad_chars]) and p != '' and p != '\n' and p[0] != '(']
    return new_lines

def clean_open_subtitles(lines):
    """ For open_subtitles, we lowercase, remove subtitle dashes and fix the lowercase 'l' problem. We also join every 5 lines. """
    punctuation = ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '“', '”', '—', '–', ' ', '\n']
    new_lines = []
    joined = []
    count = 0
    for line in lines:
        new_line = line.lower()
        # Skip music lines
        if '♪' in new_line or '[' in new_line or ']' in new_line or '‎' in new_line:
            continue
        if new_line[0:2] in ["- ", "– ", "— "]:
            new_line = new_line[2:]
        if new_line[0] in ["-", "–", "—"]:
            new_line = new_line[1:]
        new_line = ' ' + new_line
        for punct in punctuation:
            new_line = new_line.replace(f" l{punct}", f" i{punct}")
            new_line = new_line.replace(f" lm{punct}", f" im{punct}")
            new_line = new_line.replace(f" lf{punct}", f" if{punct}")
        new_line = new_line.replace(' lc', ' ic')
        new_line = new_line.replace(' ld', ' id')
        new_line = new_line.replace(' lj', ' i j')
        new_line = new_line.replace(' ln', ' in')
        new_line = new_line.replace(' lp', ' ip')
        new_line = new_line.replace(' lr', ' ir')
        new_line = new_line.replace(' ls', ' is')
        new_line = new_line.replace(' isd', ' lsd')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lv', ' iv')
        if new_line.strip() != '':
            joined.append(new_line.strip())
            count += 1
            if count % 5 == 0:
                new_lines.append(" ".join(joined) + '\n')
                joined = []
    return new_lines

def clean_qed(lines):
    """ For qed, we lowercase and normalise punctuation, remove words contained in parentheses,
    remove lines that are just character's names and fix the lowercase 'l' problem. We also join every 5 lines. """

    new_lines = []
    count = 0
    joined = []
    for line in lines:
        # Before lowercasing, check if the words in the line are uppercase containing lowercase 'l' instead of 'I' and fix accordingly
        words = line.split()
        for i, word in enumerate(words):
            if word.replace('l','I').isupper() and 'l' in word and word != 'I\'ll':
                words[i] = word.replace('l', 'I')
        new_line = ' '.join(words).lower()
        new_line = new_line.replace(' lc', ' ic')
        new_line = new_line.replace(' ld', ' id')
        new_line = new_line.replace(' lj', ' i j')
        new_line = new_line.replace(' ln', ' in')
        new_line = new_line.replace(' lp', ' ip')
        new_line = new_line.replace(' lr', ' ir')
        new_line = new_line.replace(' ls', ' is')
        new_line = new_line.replace(' isd', ' lsd')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lt', ' it')
        new_line = new_line.replace(' lv', ' iv')
        new_line = new_line.replace('&amp;gt;', '')
        new_line = new_line.replace('&amp;lt;i', '')
        new_line = new_line.replace('&amp;lt;/i', '')
        new_line = new_line.replace('&amp;gt;i', '')
        new_line = new_line.replace('&amp;gt;/i', '')
        new_line = new_line.replace('&amp;gt', '')
        new_line = new_line.replace('&amp;lt', '')
        new_line = new_line.replace('&amp;amp;', '')

        # Skip lines that are just character names, e.g. "AMY GOODMAN:"
        if len(new_line.strip()) < 1 or (len(words) <= 3 and new_line.strip()[-1] == ':'):
            continue

        # Remove subtitle dashes
        if new_line[0:2] == "- ":
            new_line = new_line[2:]
        if new_line[0] == "-":
            new_line = new_line[1:]

        # Remove substrings contained within circular or square parantheses (screen descriptions)
        pattern = r'\([^)]*\)'
        new_line = re.sub(pattern, '', new_line)
        pattern = r'\[[^)]*\]'
        new_line = re.sub(pattern, '', new_line)
        new_line = new_line.replace('"', '\'')

        # Remove strange characters
        new_line = new_line.replace('#','')
        new_line = new_line.replace('*','')

        new_line = new_line.strip()
        if new_line != "":
            joined.append(new_line)
            count += 1
            if count % 5 == 0:
                new_lines.append(" ".join(joined) + '\n')
                joined = []
    return new_lines

def clean_simple_wikipedia(lines):
    """ For simple_wikipedia, we lowercase, remove empty lines and article names."""
    new_lines = []
    next_line_is_article_name = False
    for line in lines:
        if next_line_is_article_name:
            next_line_is_article_name = False
            continue
        if line.strip() == "":
            next_line_is_article_name = True
            continue
        if len(line.split()) > 2:
            new_lines.append(line.lower())
    return new_lines

def clean_switchboard(lines):
    """ For switchboard, we lowercase and join every 5 lines. """
    new_lines = []
    count = 0
    joined = []
    for line in lines:
        new_line = line.lower().strip()
        joined.append(new_line)
        count += 1
        if count % 5 == 0:
            new_lines.append(" ".join(joined) + '\n')
            joined = []
    return new_lines

def clean_wikipedia(lines):
    """ For wikipedia, we lowercase and remove empty lines and article names.
     We also remove lines that seem to be figure names or table entries. """
    new_lines = []
    for line in lines:
        new_line = line.strip()
        words = new_line.split()
        
        # Remove empty lines and article names
        if new_line == "":
            continue
        if new_line[0] == "=" and new_line[-1] == "=":
            continue

        # Filter out lines that seem to be figure names or table entries
        all_numeric = True
        all_uppercase = True
        for word in words:
            if not word.isnumeric():
                all_numeric = False
            if not word[0].isupper():
                all_uppercase = False
        if all_numeric or all_uppercase:
            continue
    
        new_lines.append(new_line.lower().strip() + '\n')
    return new_lines


# === Cleaning logic (you already have this)
def clean_data_by_source(lines, source_name):
    def normalize_typography(text):
        text = unicodedata.normalize('NFKC', text)
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'([?.!]){2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([?.!,;:\'\"\)\]\}])', r'\1', text)
        text = re.sub(r'([(\[\{])\s', r'\1', text)
        return text.strip()

    def is_extraneous_line(line):
        line = line.strip()
        if re.match(r'^(page\s*)?\d+$', line.lower()):
            return True
        if re.match(r'^\[\d+\]', line):
            return True
        if len(re.findall(r'[\t ]', line)) > 4 and len(re.findall(r'\d', line)) > 2:
            return True
        if re.match(r'^[\(\[]\s*\w+\s*[\)\]]$', line.lower()):
            return True
        if len(line.split()) <= 1:
            return True
        return False

    filtered_lines = [normalize_typography(line) for line in lines if not is_extraneous_line(line)]

    if source_name == 'open_subtitles':
        return clean_open_subtitles(filtered_lines)
    elif source_name == 'bnc_spoken':
        return clean_bnc_spoken(filtered_lines)
    elif source_name == 'gutenberg':
        return clean_gutenberg(filtered_lines)
    elif source_name == 'simple_wikipedia':
        return clean_simple_wikipedia(filtered_lines)
    else:
        raise ValueError(f"Unsupported source_name '{source_name}'.")

# === GENEREATING KIDLM_70M.TXT


# === Load and clean each file
def load_and_clean_file(path, source_name):
    with open(path, 'r') as f:
        lines = f.readlines()
    return clean_data_by_source(lines, source_name)

# === Load + clean OpenSubtitles
open_subs_path = [TO DO] - INSERT LINK TO OpenSubtitles
open_subs_clean = load_and_clean_file(open_subs_path, "open_subtitles")

# === Load + clean BNC Spoken
bnc_path = [TO DO] - INSERT LINK TO BNC Path
bnc_clean = load_and_clean_file(bnc_path, "bnc_spoken")

# === Load + clean Switchboard
switchboard_path = [TO DO] - INSERT LINK TO Switchboard
switchboard_clean = load_and_clean_file(switchboard_path, "bnc_spoken")  # or your custom switchboard cleaner

# === Combine all cleaned lines
all_clean_lines = open_subs_clean + bnc_clean + switchboard_clean

# === Apply post-cleaning concatenation
# You already have this function presumably defined
# def clean_and_concatenate_lines(lines, concat_size=5, lowercase=False, skip_concat_for_bnc=False, is_bnc=False)

concatenated_lines = clean_and_concatenate_lines(
    all_clean_lines,
    concat_size=5,
    lowercase=True,
    skip_concat_for_bnc=True,
    is_bnc=False
)

# === Append KIDLM lines - convert the KidLM train.json --> text file and then append these new files --> generate a training txt file. 

#with open("/Users/suchirsalhan/Documents/PHD/PHD/babylm/data/raw/kidlm/kidlm.txt", "r") as kidlm_f:
#    kidlm_lines = [line.strip() for line in kidlm_f if line.strip() != ""]
#final_output_lines = concatenated_lines + kidlm_lines

# === Save to final output file
final_output_path = "raw/kidlm/kidlm-70M.txt - REPLACE PATH"
with open(final_output_path, "w") as f:
    for line in final_output_lines:
        f.write(line.strip() + "\n")

print(f"Saved final cleaned data to {final_output_path}")
