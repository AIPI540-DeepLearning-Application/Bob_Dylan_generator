#!/bin/bash

poems_title=(
    "Shadows in the Moonlight"
    "Echoes of Forgotten Dreams"
    "Tides of Change"
    "Beneath the Velvet Sky"
    "Dancing with the Stars"
    "In the Heart of the Storm"
    "Serenade of the Sea"
    "Flames of Desire"
    "Through the Eyes of Time"
    "Whispers Among the Ruins"
    "Glimpses of Eternity"
    "Rhythms of the Rain"
    "Veils of Mist"
    "Embers of a Fading Sun"
    "Bridges Over Silent Waters"
    "Silent Whispers of the Night"
    "Melodies in the Twilight"
    "Footprints in the Sand"
    "Whispers of the Wind"
    "Mysteries of the Cosmos"
    "Songs of the Soul"
    "Harmony of the Universe"
    "Whispers of the Heart"
    "Stardust Serenade"
    "Journey to the Unknown"
    "Silent Echoes of Love"
    "Eternal Flames of Passion"
    "Chasing the Horizon"
    "Whispers of the Enchanted Forest"
    "Sailing through Stardust"
    "Moonlit Reflections"
    "Whispers of the Winter's Eve"
    "Whispers of the Autumn Leaves"
    "Whispers of the Spring Breeze"
    "Whispers of the Summer Rain"
    "Whispers of the Frozen Lake"
    "Whispers of the Mountain Peaks"
    "Whispers of the Flowing River"
    "Whispers of the Ocean Waves"
    "Whispers of the Desert Sands"
    "Whispers of the Starry Night"
    "Whispers of the Midnight Sky"
    "Whispers of the Distant Galaxies"
    "Whispers of the Nebulae"
    "Whispers of the Celestial Symphony"
    "Whispers of the Luminous Moon"
    "Whispers of the Shimmering Stars"
    "Whispers of the Twilight Sky"
    "Whispers of the Enchanted Garden"
    "Whispers of the Secret Garden"
)


model_hf_arr=(
    "Pot-l/llama-7b-bobdylan"
    "meta-llama/Llama-2-7b-chat-hf"
)

for model_hf in "${model_hf_arr[@]}"
do
    for title in "${poems_title[@]}"
    do
        python test_GPU_inference.py "$model_hf" "$title"
    done
done