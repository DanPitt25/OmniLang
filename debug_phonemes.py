#!/usr/bin/env python3
import sys
sys.path.insert(0, './IMS-Toucan')

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# Test the TextFrontend directly
print("Testing TextFrontend with input_phonemes=True")
tf = ArticulatoryCombinedTextFrontend(language="vec")

# Test with and without input_phonemes
test_text = "θat is miːnə riːtʃə"

print(f"\nInput text: {test_text}")
print("\n1. With input_phonemes=False (normal text processing):")
try:
    result1 = tf.string_to_tensor(test_text, view=True, input_phonemes=False)
    print(f"Tensor shape: {result1.shape}")
except Exception as e:
    print(f"Error: {e}")

print("\n2. With input_phonemes=True (should bypass phonemizer):")
try:
    result2 = tf.string_to_tensor(test_text, view=True, input_phonemes=True)
    print(f"Tensor shape: {result2.shape}")
except Exception as e:
    print(f"Error: {e}")