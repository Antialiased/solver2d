// SPDX-FileCopyrightText: 2026 solver2d deformable experiments
// SPDX-License-Identifier: MIT

#include "sample.h"
#include "settings.h"

#include "solver2d/geometry.h"
#include "solver2d/math.h"
#include "solver2d/solver2d.h"

#include <stdio.h>

// Rigid circle dropped onto a static ground segment with restitution = 1.
// Tracks successive apex heights to report energy preservation per bounce,
// since at the apex KE = 0 so E_tot = m*g*h and h_n/h_0 is the energy ratio.
class EnergyBounce : public Sample
{
public:
	EnergyBounce(const Settings& settings, s2SolverType solverType)
		: Sample(settings, solverType)
	{
		if (settings.restart == false)
		{
			g_camera.m_center = {0.0f, 3.0f};
			g_camera.m_zoom = 0.25f;
		}

		// Ground segment at y = 0
		s2BodyId groundId = s2CreateBody(m_worldId, &s2_defaultBodyDef);
		s2Segment segment = {{-20.0f, 0.0f}, {20.0f, 0.0f}};
		s2ShapeDef groundShape = s2_defaultShapeDef;
		groundShape.friction = 0.0f;
		groundShape.restitution = 1.0f;
		s2CreateSegmentShape(groundId, &groundShape, &segment);

		// Dynamic ball dropped from rest
		s2BodyDef bodyDef = s2_defaultBodyDef;
		bodyDef.type = s2_dynamicBody;
		bodyDef.position = {0.0f, m_h0 + m_radius};
		m_ballId = s2CreateBody(m_worldId, &bodyDef);

		s2ShapeDef shapeDef = s2_defaultShapeDef;
		shapeDef.friction = 0.0f;
		shapeDef.restitution = 1.0f;
		shapeDef.density = 1.0f;
		s2Circle circle = {{0.0f, 0.0f}, m_radius};
		s2CreateCircleShape(m_ballId, &shapeDef, &circle);

		m_yPrev = m_h0;
		m_yPrevPrev = m_h0;
		m_apexCount = 0;
		m_lastApex = m_h0;
		m_firstApex = m_h0;
	}

	virtual void Step(Settings& settings, s2Color bodyColor) override
	{
		Sample::Step(settings, bodyColor);

		s2Vec2 p = s2Body_GetPosition(m_ballId);
		float y = p.y - m_radius; // height of the lowest point above the floor

		// Local-max detection on a 3-sample window: y_prev is an apex
		// if y_prevPrev < y_prev and y_prev > y.
		if (m_yPrevPrev < m_yPrev && m_yPrev > y && m_stepCount > 2)
		{
			m_lastApex = m_yPrev;
			if (m_apexCount == 0)
			{
				m_firstApex = m_yPrev;
			}
			m_apexCount += 1;
		}
		m_yPrevPrev = m_yPrev;
		m_yPrev = y;

		float ratio = (m_firstApex > 0.0f) ? (m_lastApex / m_firstApex) : 1.0f;

		g_draw.DrawString(5, settings.textLine, "restitution = 1.0, e_target = 1.000");
		settings.textLine += settings.textIncrement;
		g_draw.DrawString(5, settings.textLine, "apex #%d  h_last = %.4f  h_last/h_first = %.4f",
			m_apexCount, m_lastApex, ratio);
		settings.textLine += settings.textIncrement;
		g_draw.DrawString(5, settings.textLine, "h_initial = %.4f  y_now = %.4f", m_h0, y);
		settings.textLine += settings.textIncrement;
	}

	static Sample* Create(const Settings& settings, s2SolverType solverType)
	{
		return new EnergyBounce(settings, solverType);
	}

	s2BodyId m_ballId;
	static constexpr float m_h0 = 5.0f;
	static constexpr float m_radius = 0.5f;
	float m_yPrev;
	float m_yPrevPrev;
	float m_lastApex;
	float m_firstApex;
	int m_apexCount;
};

static int sampleEnergyBounce = RegisterSample("Contact", "Energy Bounce", EnergyBounce::Create);
