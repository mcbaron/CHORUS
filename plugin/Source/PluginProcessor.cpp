#include "PluginProcessor.h"
#include "PluginEditor.h"

extern "C" void* chorus_create(unsigned int sample_rate);
extern "C" void chorus_destroy(void* handle);
extern "C" int chorus_process_interleaved(void* handle, const float* input, float* output, size_t frames);

CHORUSAudioProcessor::CHORUSAudioProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                      .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

CHORUSAudioProcessor::~CHORUSAudioProcessor()
{
    if (chorusHandle != nullptr)
        chorus_destroy(chorusHandle);
}

void CHORUSAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    if (chorusHandle != nullptr)
        chorus_destroy(chorusHandle);
    chorusHandle = chorus_create(static_cast<unsigned int>(sampleRate));
    const size_t capacity = static_cast<size_t>(samplesPerBlock) * 2 + 32;
    interleavedBuffer.resize(capacity);
    processedBuffer.resize(capacity);
}

void CHORUSAudioProcessor::releaseResources() {}

bool CHORUSAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo()
        && layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void CHORUSAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    if (chorusHandle == nullptr || buffer.getNumChannels() < 2)
    {
        buffer.clear();
        return;
    }
    const int frames = buffer.getNumSamples();
    const size_t requiredSamples = static_cast<size_t>(frames) * 2;
    if (requiredSamples > interleavedBuffer.size())
    {
        interleavedBuffer.resize(requiredSamples);
        processedBuffer.resize(requiredSamples);
    }
    for (int i = 0; i < frames; ++i)
    {
        interleavedBuffer[static_cast<size_t>(i) * 2] = buffer.getSample(0, i);
        interleavedBuffer[static_cast<size_t>(i) * 2 + 1] = buffer.getSample(1, i);
    }
    const int status = chorus_process_interleaved(chorusHandle, interleavedBuffer.data(), processedBuffer.data(), static_cast<size_t>(frames));
    if (status == 0)
    {
        for (int i = 0; i < frames; ++i)
        {
            buffer.setSample(0, i, processedBuffer[static_cast<size_t>(i) * 2]);
            buffer.setSample(1, i, processedBuffer[static_cast<size_t>(i) * 2 + 1]);
        }
    }
    else
    {
        buffer.clear();
    }
}

juce::AudioProcessorEditor* CHORUSAudioProcessor::createEditor() { return new CHORUSAudioProcessorEditor(*this); }
bool CHORUSAudioProcessor::hasEditor() const { return true; }
const juce::String CHORUSAudioProcessor::getName() const { return "CHORUS"; }
bool CHORUSAudioProcessor::acceptsMidi() const { return false; }
bool CHORUSAudioProcessor::producesMidi() const { return false; }
double CHORUSAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int CHORUSAudioProcessor::getNumPrograms() { return 1; }
int CHORUSAudioProcessor::getCurrentProgram() { return 0; }
void CHORUSAudioProcessor::setCurrentProgram(int) {}
const juce::String CHORUSAudioProcessor::getProgramName(int) { return {}; }
void CHORUSAudioProcessor::changeProgramName(int, const juce::String&) {}
void CHORUSAudioProcessor::getStateInformation(juce::MemoryBlock& destData) { destData.setSize(0); }
void CHORUSAudioProcessor::setStateInformation(const void*, int) {}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CHORUSAudioProcessor();
}
