#include <JuceHeader.h>
#include "../Source/PluginProcessor.h"

int main()
{
    CHORUSAudioProcessor processor;
    processor.prepareToPlay(48000.0, 512);
    juce::AudioBuffer<float> buffer(2, 512);
    juce::MidiBuffer midi;
    buffer.clear();
    processor.processBlock(buffer, midi);
    juce::MemoryBlock state;
    processor.getStateInformation(state);
    if (state.getSize() != 0)
        return 1;
    processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    return 0;
}
