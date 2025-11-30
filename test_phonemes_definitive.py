#!/usr/bin/env python3
import sys
sys.path.insert(0, './IMS-Toucan')

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# Test the core function that input_is_phones controls
tf = ArticulatoryCombinedTextFrontend(language="vec")

test_text = "θat is miːnə riːtʃə"

print("=== PHONEMIZER ON (input_phonemes=False) ===")
result1 = tf.string_to_tensor(test_text, view=True, input_phonemes=False)
print(f"Shape: {result1.shape}\n")

print("=== PHONEMIZER OFF (input_phonemes=True) ===") 
result2 = tf.string_to_tensor(test_text, view=True, input_phonemes=True)
print(f"Shape: {result2.shape}\n")

print("=== COMPARISON ===")
print(f"Shapes are different: {result1.shape != result2.shape}")
print(f"Tensors are different: {not result1.equal(result2) if result1.shape == result2.shape else 'N/A (different shapes)'}")