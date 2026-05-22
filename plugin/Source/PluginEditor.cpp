#include "PluginEditor.h"

CHORUSAudioProcessorEditor::CHORUSAudioProcessorEditor(CHORUSAudioProcessor& processorRef)
    : AudioProcessorEditor(&processorRef), processor(processorRef)
{
    setSize(420, 180);
}

void CHORUSAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.setFont(24.0f);
    g.drawFittedText("CHORUS", getLocalBounds(), juce::Justification::centred, 1);
}

void CHORUSAudioProcessorEditor::resized() {}
