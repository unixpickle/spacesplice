package spacesplice

import "github.com/unixpickle/serializer"

const (
	serializerPrefix          = "github.com/unixpickle/spacesplice."
	serializerTypeMarkov      = serializerPrefix + "Markov"
	serializerTypeDictionary  = serializerPrefix + "Dictionary"
	serializerTypeForest      = serializerPrefix + "Forest"
	serializerTypeRNN         = serializerPrefix + "RNN"
	serializerTypeBoostStumps = serializerPrefix + "BoostStumps"
)

func init() {
	serializer.RegisterTypedDeserializer(serializerTypeMarkov, DeserializeMarkov)
	serializer.RegisterTypedDeserializer(serializerTypeDictionary, DeserializeDictionary)
	serializer.RegisterTypedDeserializer(serializerTypeForest, DeserializeForest)
	serializer.RegisterTypedDeserializer(serializerTypeRNN, DeserializeRNN)
	serializer.RegisterTypedDeserializer(serializerTypeBoostStumps, DeserializeBoostStumps)
}
