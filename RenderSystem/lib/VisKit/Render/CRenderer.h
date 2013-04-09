#ifndef _CRENDERER_H_
#define _CRENDERER_H_

#include "CObject.h"

class CShader;
class CData;

class CRenderer : public CObject
{
public:
	CRenderer();
	~CRenderer();

	virtual void	init()=0;
	virtual void	render()=0;

	void		setShader(CShader*);
	void		setData(CData*);

protected:
	CShader		*m_shader;
	CData		*m_data;
};


#endif

