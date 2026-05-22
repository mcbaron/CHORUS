#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class CHORUSAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit CHORUSAudioProcessorEditor(CHORUSAudioProcessor&);
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    CHORUSAudioProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CHORUSAudioProcessorEditor)
};
