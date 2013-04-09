#include "CRenderer.h"
#include "CShader.h"

CRenderer::CRenderer()
{
}
CRenderer::~CRenderer()
{
}
void CRenderer::setShader(CShader *s)
{
	m_shader = s;
	m_shader->setRenderer(this);
}
void CRenderer::setData(CData *d)
{
	m_data = d;
}