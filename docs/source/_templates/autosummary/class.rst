{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}

   {% set public_methods = [] %}
   {% for item in methods %}
   {% if not (item == "__init__"
              or item in inherited_members
              or item == "cross_entropy"
              or item == "kl_divergence") %}
   {% set _ = public_methods.append(item) %}
   {% endif %}
   {% endfor %}

   {% if public_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :nosignatures:

      {% for item in public_methods %}
      ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}

   {% endblock %}

   {% block attributes %}

   {% set public_attributes = [] %}
   {% for item in attributes %}
   {% if item not in inherited_members %}
   {% set _ = public_attributes.append(item) %}
   {% endif %}
   {% endfor %}

   {% if public_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: autosummary/attribute.rst
      :toctree:

      {% for item in public_attributes %}
      ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}

   {% endblock %}