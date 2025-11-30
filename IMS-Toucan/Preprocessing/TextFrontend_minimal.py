#!/usr/bin/env python3
"""
Minimal TextFrontend for IPA passthrough only
"""

import re
import torch
from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_feature_to_index_lookup

class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 language,
                 use_explicit_eos=True,
                 use_lexical_stress=True,
                 silent=True,
                 add_silence_to_end=True,
                 use_word_boundaries=True,
                 device="cpu"):
        """
        Minimal frontend for IPA passthrough only
        """
        self.language = language
        self.use_explicit_eos = use_explicit_eos
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.use_word_boundaries = use_word_boundaries
        
        # No phonemizers - just pass through IPA
        self.g2p_lang = language
        self.expand_abbreviations = lambda x: x

        # Initialize feature mappings
        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_feature_to_index_lookup()
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False):
        """
        Convert text to tensor. If input_phonemes=True, bypass phonemizer.
        """
        if input_phonemes:
            # Direct IPA passthrough
            phones = text
        else:
            # For compatibility, still do some basic cleanup but no phonemization
            phones = text
            
        phones = phones.replace("ɚ", "ə").replace("ᵻ", "ɨ")
        if view:
            print("Phonemes: \n{}\n".format(phones))
            
        phones_vector = list()
        stressed_flag = False
        
        # Process each character
        for char in phones:
            if char == "ˈ":
                stressed_flag = True
                continue
            if char == "ˌ":
                stressed_flag = True  
                continue
                
            if char in self.phone_to_id:
                if stressed_flag and self.use_stress:
                    phone_id = self.phone_to_id[char + "ˈ"] if char + "ˈ" in self.phone_to_id else self.phone_to_id[char]
                else:
                    phone_id = self.phone_to_id[char]
                phones_vector.append(self.phone_to_vector[phone_id])
                stressed_flag = False
            elif handle_missing and char != " " and char != "~":
                # Skip unknown characters silently
                continue
                
        if self.use_explicit_eos and self.add_silence_to_end:
            phones_vector.append(self.phone_to_vector[self.phone_to_id["#"]])
            
        return torch.Tensor(phones_vector).to(device)

    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False):
        """
        For compatibility - just return the input text
        """
        if text == "":
            return ""
        return text

    def phoneme_list_to_id_list(self, phoneme_list):
        """
        Convert phoneme list to ID list
        """
        id_list = []
        for phoneme in phoneme_list:
            if phoneme in self.phone_to_id:
                id_list.append(self.phone_to_id[phoneme])
        return id_list