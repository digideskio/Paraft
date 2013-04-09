varying vec3  textureCoords;

void main()
{
	gl_TexCoord[0].xyz = gl_Vertex.xyz;//(gl_TextureMatrix[0] * gl_MultiTexCoord0).xyz;
	gl_Position = ftransform();	
	textureCoords = gl_Vertex.xyz;
}
