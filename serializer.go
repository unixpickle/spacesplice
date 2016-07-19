package spacesplice

import "github.com/unixpickle/serializer"

const (
	serializerPrefix     = "github.com/unixpickle/spacesplice/"
	serializerTypeMarkov = serializerPrefix + "Markov"
)

func init() {
	serializer.RegisterTypedDeserializer(serializerTypeMarkov, DeserializeMarkov)
}
